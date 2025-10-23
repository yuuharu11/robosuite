import copy
import torch
import torch.nn as nn
from typing import Any, Dict, List

import src.utils as utils
from src.utils import registry

class PNN(nn.Module):
    def __init__(self, base_model_config: Dict[str, Any], d_output: int, task_id: int = 0):
        super().__init__()
        self.base_model_config = copy.deepcopy(base_model_config)
        self.d_output = d_output
        self.task_id = task_id
        self.task_to_col: Dict[int, int] = {}

        # 各タスクのモデル（列）を格納
        self.columns: nn.ModuleList = nn.ModuleList()
        # 列ごと・層ごとの横方向接続パラメータ
        self.laterals: nn.ModuleList = nn.ModuleList()

    def _build_column(self) -> nn.Module:
        config = copy.deepcopy(self.base_model_config)
        if config.get('_name_') == 'pnn':
            config['_name_'] = 'base'
        col = utils.instantiate(registry.model, config)
        return col

    def add_column(self, task_id: int):
        if task_id in self.task_to_col:
            print(f"Task {task_id} already has a column.")
            return
        col = self._build_column()

        units_config = self.base_model_config.get("layer", {}).get("units", [])
        output_units = next((u.get("output_units") for u in units_config if "output_units" in u), None)
        if output_units is None:
            raise ValueError("output_units が設定にありません")
        col.output_layer = nn.Linear(output_units, self.d_output)

        self.columns.append(col)

        """
        # 横方向接続用パラメータを初期化
        adapters = nn.ModuleList()
        if len(self.columns) > 1:
            for prev_col in self.columns[:-1]:
                lateral_per_layer = nn.ModuleList()
                for l_new, l_prev in zip(col.layers, prev_col.layers):
                    lateral = nn.Linear(l_prev.out_features, l_new.out_features, bias=False)
                    nn.init.zeros_(lateral.weight)
                    lateral_per_layer.append(lateral)
                adapters.append(lateral_per_layer)
        self.laterals.append(adapters)
        """

        col_idx = len(self.columns) - 1
        self.task_to_col[task_id] = col_idx

    def freeze_previous_columns(self):
        num_columns = len(self.columns)
        if num_columns <= 1:
            return
        for i in range(num_columns - 1):
            for param in self.columns[i].parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor, **kwargs):
        if self.task_id not in self.task_to_col:
            raise ValueError(f"Task {self.task_id} に対応するカラムがありません")
        col_idx = self.task_to_col[self.task_id]
        target_column = self.columns[col_idx]

        all_states = []

        # 横方向接続付き forward
        for layer_idx, layer in enumerate(target_column.layers):
            result = layer(x)
            if isinstance(result, tuple):
                x_new, state_new = result
            else:
                x_new, state_new = result, None
            all_states.append(state_new)
            lateral_sum = 0.0
            for prev_idx, prev_col in enumerate(self.columns[:-1]):
                prev_out = prev_col.layers[layer_idx](x)
                lateral_layer = self.laterals[col_idx][prev_idx][layer_idx]
                lateral_sum += lateral_layer(prev_out)
            x = x_new + lateral_sum

        # 出力層
        output = target_column.output_layer(x)

        return output, all_states
