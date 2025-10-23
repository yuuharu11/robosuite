from __future__ import print_function

import collections

import numpy as np

import torch
import torch.nn as nn


class SparsePruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, prune_perc, previous_masks, train_bias, train_bn):
        self.model = model
        self.prune_perc = prune_perc
        self.train_bias = train_bias
        self.train_bn = train_bn

        self.current_masks = None
        self.previous_masks = previous_masks
        if previous_masks and len(previous_masks) > 0:
            valid_key = list(previous_masks.keys())[0]
            self.current_dataset_idx = previous_masks[valid_key].max()
        else:
            self.current_dataset_idx = 0

    def pruning_mask(self, weights, available_mask, layer_idx):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        # 1. プルニング対象となる「利用可能な重み」の絶対値を取得
        available_weights = weights[available_mask.eq(1)]
        if available_weights.numel() == 0:
            # 利用可能な重みがなければ、空のマスクを返す
            return torch.zeros_like(weights, dtype=torch.int).cuda()

        abs_weights = available_weights.abs()
        
        # 2. 枝刈りする重みの数を計算
        num_to_prune = int(self.prune_perc * abs_weights.numel())

        # 3. 枝刈りの閾値（カットオフ値）を決定
        #    重要度が低い（＝絶対値が小さい）ものから数えて、num_to_prune番目の値
        cutoff_value = abs_weights.view(-1).kthvalue(num_to_prune)[0]

        # 4. 今回のタスクで「獲得」する重みのマスクを新規に作成
        #    (利用可能 かつ 絶対値がカットオフ値より大きいもの)
        new_task_mask = available_mask * weights.abs().gt(cutoff_value)
        
        return new_task_mask.int()

    def prune(self):
        """Gets pruning mask for each layer, based on previous_masks.
        Sets the self.current_masks to the computed pruning masks.
        """
        assert not self.current_masks, 'Current mask is not empty? Pruning twice?'
        self.current_masks = {}

        # ---- タスク1: 既存マスクが無い場合 ----
        if not self.previous_masks or len(self.previous_masks) == 0:
            print(f'Pruning for dataset idx: {self.current_dataset_idx}')
            print("[green]Pruning first task weights[/green]")
            mask_layer_idx = 0
            for param_name, param in self.model.named_parameters():
                print(param_name, param.size())
                if ('weight' in param_name or '_w' in param_name) and param.numel() > 0:
                    print(param_name, param.size())
                    # 全重みをタスク0として扱い pruning
                    available_mask = torch.ones_like(param.data, dtype=torch.int).cuda()

                    # magnitude pruning 実行
                    pruned_mask = self.pruning_mask(
                        param.data, available_mask, mask_layer_idx
                    )

                    self.current_masks[param_name] = pruned_mask.cuda()

                    # pruneした重みをゼロ化
                    param.data[self.current_masks[param_name].eq(0)] = 0.0
                    mask_layer_idx += 1
            
            return

        # ---- タスク1以降 ----
        else:
            print("[green]Pruning subsequent task weights[/green]")
            self.current_dataset_idx += 1
            mask_layer_idx = 0
            print(f'Pruning for dataset idx: {self.current_dataset_idx}')
            for param_name, param in self.model.named_parameters():
                if 'weight' in param_name or '_w' in param_name:
                    
                    # 1. 過去のタスクで使われている重みのマスクを取得
                    previous_mask = self.previous_masks[param_name].cuda()
                    
                    # 2. 今回のタスクで利用可能な重み（過去に使われていない重み）を特定
                    available_mask = previous_mask.eq(0).int().cuda()
                    
                    print(f'Pruning layer "{param_name}" ({mask_layer_idx}). Available weights: {available_mask.sum()}')

                    # 3. 利用可能な重みの中だけで、さらにプルニングを実行
                    pruned_available_mask = self.pruning_mask(
                        param.data, available_mask, mask_layer_idx
                    )
                    
                    # 4. 新しいマスクは、「過去のマスク」＋「今回新たに獲得したマスク」の合計
                    self.current_masks[param_name] = previous_mask + pruned_available_mask

                    # 5. 最新のマスクをモデルの重みに適用
                    param.data[self.current_masks[param_name].eq(0)] = 0.0
                    mask_layer_idx += 1
            
            return
        
    def make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.current_masks

        for param_name, module in self.model.named_modules():
            if 'weight' in param_name or '_w' in param_name:
                layer_mask = self.current_masks[param_name]

                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(
                        self.current_dataset_idx)] = 0
                    if not self.train_bias:
                        # Biases are fixed.
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)
            elif 'BatchNorm' in str(type(module)):
                # Set grads of batchnorm params to 0.
                if not self.train_bn:
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.current_masks

        for param_name, module in self.model.named_modules():
            if 'weight' in param_name or '_w' in param_name:
                layer_mask = self.current_masks[param_name]
                module.weight.data[layer_mask.eq(0)] = 0.0

    def apply_mask(self, dataset_idx):
        """To be done to retrieve weights just for a particular dataset."""
        for param_name, module in self.model.named_modules():
            if 'weight' in param_name or '_w' in param_name:
                weight = module.weight.data
                mask = self.previous_masks[param_name].cuda()
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(dataset_idx)] = 0.0

    def restore_biases(self, biases):
        """Use the given biases to replace existing biases."""
        for param_name, module in self.model.named_modules():
            if 'weight' in param_name or '_w' in param_name:
                if module.bias is not None:
                    module.bias.data.copy_(biases[param_name])

    def get_biases(self):
        """Gets a copy of the current biases."""
        biases = {}
        for param_name, module in self.model.named_modules():
            if 'weight' in param_name or '_w' in param_name:
                if module.bias is not None:
                    biases[param_name] = module.bias.data.clone()
        return biases

    def make_finetuning_mask(self):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        if not self.previous_masks or len(self.previous_masks) == 0:
        # 新しいタスク用のマスクを初期化
            self.current_masks = {}
            self.current_dataset_idx = 0
            return

        self.current_dataset_idx += 1

        for param_name, module in self.model.named_modules():
            if 'weight' in param_name or '_w' in param_name:
                mask = self.previous_masks[param_name]
                mask[mask.eq(0)] = self.current_dataset_idx

        self.current_masks = self.previous_masks
