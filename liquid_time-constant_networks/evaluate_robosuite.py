#!/usr/bin/env python
"""
Robosuite rollout evaluator using trained NCP/LTC model weights.
Measures average reward, steps, inference time, FPS, success rate, and GPU memory.
"""
import argparse
import time
import numpy as np
import torch
import robosuite as suite
from train import SequenceLightningModule

SUCCESS_THRESHOLD = 50.0  # タスク依存で調整

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts")
    parser.add_argument("--max_steps", type=int, default=400, help="Max steps per rollout")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model")
    parser.add_argument("--save_video", action="store_true", help="Save video of first rollout")
    parser.add_argument("--render", action="store_true", help="Render environment")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load model
    model = SequenceLightningModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(args.device)

    # Create environment
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        use_object_obs=True,
        horizon=args.max_steps,
        control_freq=100
    )

    rollout_rewards = []
    rollout_steps = []
    inference_times = []
    max_memory = []

    print(f"✅ Starting {args.num_rollouts} rollouts")
    seq_len = 10

    for rollout_idx in range(args.num_rollouts):
        obs_history = []
        obs = env.reset()
        total_reward = 0.0
        state = None

        for step in range(args.max_steps):
            obs_vec = np.concatenate([
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"],
                obs["robot0_gripper_qpos"],
                obs["object-state"]
            ]).astype(np.float32)

            obs_history.append(obs_vec)
            if len(obs_history) < seq_len:
                seq = [obs_history[0]] * (seq_len - len(obs_history)) + obs_history
            else:
                seq = obs_history[-seq_len:]

            obs_tensor = torch.from_numpy(np.stack(seq)).unsqueeze(0).to(args.device)

            # 推論時間 & GPU メモリ計測
            torch.cuda.reset_peak_memory_stats(args.device)
            start_time = time.time()
            with torch.no_grad():
                action_seq, state = model.rollout_forward(obs_tensor, state)
                action = action_seq[0, -1, :].cpu().numpy()
            elapsed = time.time() - start_time
            inference_times.append(elapsed)
            max_memory.append(torch.cuda.max_memory_allocated(args.device) / (1024 ** 2))  # MB換算

            action = np.clip(action, -1.0, 1.0)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if args.render:
                env.render()

            if done:
                break

        rollout_rewards.append(total_reward)
        rollout_steps.append(step + 1)

        print(f"Rollout {rollout_idx+1}/{args.num_rollouts}: Reward={total_reward:.3f}, Steps={step+1}")

    env.close()

    # Metrics
    successes = [1 if r >= SUCCESS_THRESHOLD else 0 for r in rollout_rewards]
    avg_reward = np.mean(rollout_rewards)
    reward_std = np.std(rollout_rewards)
    avg_steps = np.mean(rollout_steps)
    avg_inference_time = np.mean(inference_times)
    inference_std = np.std(inference_times)
    fps = 1.0 / avg_inference_time
    success_rate = np.mean(successes) * 100.0
    avg_memory = np.mean(max_memory)
    max_memory_peak = np.max(max_memory)

    print("\n=========================================")
    print("Evaluation Complete")
    print(f"Average Reward: {avg_reward:.3f} ± {reward_std:.3f}")
    print(f"Reward Range: [{np.min(rollout_rewards):.3f}, {np.max(rollout_rewards):.3f}]")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Inference Time per Step: {avg_inference_time*1000:.2f} ± {inference_std*1000:.2f} ms")
    print(f"Approx. FPS: {fps:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average GPU Memory Usage: {avg_memory:.2f} MB")
    print(f"Peak GPU Memory Usage: {max_memory_peak:.2f} MB")
    print("=========================================")

if __name__ == "__main__":
    main()
