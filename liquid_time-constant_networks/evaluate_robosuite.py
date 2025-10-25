"""
Evaluate trained LTC/NCP model on robosuite Lift task with rollouts.
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

# robosuite imports
try:
    import robosuite as suite
    ROBOSUITE_AVAILABLE = True
except ImportError:
    ROBOSUITE_AVAILABLE = False
    print("WARNING: robosuite not available. Install with: pip install robosuite")

from train import SequenceLightningModule


class RolloutEvaluator:
    """
    Rollout evaluator for robosuite environments.
    """
    
    def __init__(self, model, env, obs_keys, seq_len=10, device="cuda"):
        self.model = model
        self.env = env
        self.obs_keys = obs_keys
        self.seq_len = seq_len
        self.device = device
        
        # History buffer for observations (for sequence input)
        self.obs_history = []
        
        # Get normalization stats if model was trained with normalization
        self.use_norm = False
        self.obs_mean = None
        self.obs_std = None
        
    def set_normalization_stats(self, hdf5_path):
        """
        Compute normalization statistics from training data.
        """
        with h5py.File(hdf5_path, "r") as f:
            all_obs = []
            for demo_key in f["data"].keys():
                demo = f["data"][demo_key]
                obs_parts = [demo["obs"][k][:] for k in self.obs_keys]
                obs = np.concatenate(obs_parts, axis=-1)
                all_obs.append(obs)
            
            all_obs = np.concatenate(all_obs, axis=0)
            self.obs_mean = np.mean(all_obs, axis=0, keepdims=True)
            self.obs_std = np.std(all_obs, axis=0, keepdims=True) + 1e-8
            self.use_norm = True
            
            print(f"Normalization stats computed from {hdf5_path}")
            print(f"  Mean shape: {self.obs_mean.shape}, Std shape: {self.obs_std.shape}")
    
    def _get_observation_vector(self, obs_dict):
        """
        Extract observation vector from robosuite observation dict.
        """
        obs_parts = []
        for key in self.obs_keys:
            if key not in obs_dict:
                raise KeyError(f"Observation key '{key}' not found in environment obs")
            obs_parts.append(obs_dict[key])
        
        obs_vec = np.concatenate(obs_parts, axis=-1).astype(np.float32)
        
        # Apply normalization if enabled
        if self.use_norm:
            obs_vec = (obs_vec - self.obs_mean) / self.obs_std
        
        return obs_vec
    
    def reset(self):
        """Reset history buffer."""
        self.obs_history = []
    
    def get_action(self, obs_dict):
        """
        Get action from model given current observation.
        Uses history to form sequence input.
        """
        # Get observation vector
        obs_vec = self._get_observation_vector(obs_dict)
        
        # Add to history
        self.obs_history.append(obs_vec)
        
        # Keep only last seq_len observations
        if len(self.obs_history) > self.seq_len:
            self.obs_history = self.obs_history[-self.seq_len:]
        
        # Pad if we don't have enough history yet
        if len(self.obs_history) < self.seq_len:
            # Repeat first observation
            obs_seq = np.stack([self.obs_history[0]] * (self.seq_len - len(self.obs_history)) + self.obs_history, axis=0)
        else:
            obs_seq = np.stack(self.obs_history, axis=0)
        
        # Convert to tensor: (1, seq_len, obs_dim)
        obs_tensor = torch.from_numpy(obs_seq).unsqueeze(0).to(self.device)
        
        # Get action from model
        with torch.no_grad():
            action_seq = self.model(obs_tensor)  # (1, seq_len, action_dim)
            # Use last action in sequence
            action = action_seq[0, -1, :].cpu().numpy()
        
        return action
    
    def rollout(self, max_steps=500, render=False, video_writer=None):
        """
        Perform one rollout episode.
        
        Returns:
            success (bool): Whether the task was successful
            total_reward (float): Cumulative reward
            traj_len (int): Episode length
        """
        obs = self.env.reset()
        self.reset()
        
        total_reward = 0.0
        success = False
        
        for step in range(max_steps):
            # Get action from model
            action = self.get_action(obs)
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            # Check success
            if "success" in info:
                success = success or info["success"]
            
            # Render
            if render:
                if video_writer is not None:
                    frame = self.env.sim.render(width=640, height=480, camera_name="agentview")
                    video_writer.append_data(frame[::-1])  # Flip vertically
                else:
                    self.env.render()
            
            if done:
                break
        
        return success, total_reward, step + 1


def evaluate_model(
    checkpoint_path,
    config_path=None,
    num_rollouts=50,
    max_steps=500,
    render=False,
    save_video=False,
    video_path=None,
    camera_name="agentview",
    device="cuda",
):
    """
    Evaluate a trained model on robosuite Lift task.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config (if None, tries to infer from checkpoint dir)
        num_rollouts: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render (requires display)
        save_video: Whether to save video
        video_path: Path to save video
        camera_name: Camera for rendering
        device: Device to run model on
    """
    if not ROBOSUITE_AVAILABLE:
        raise ImportError("robosuite is required for rollout evaluation")
    
    # Load checkpoint and config
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if config_path is None:
        # Try to find config in checkpoint directory
        ckpt_dir = Path(checkpoint_path).parent.parent
        config_path = ckpt_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}. Please specify --config_path")
    
    print(f"Loading config: {config_path}")
    config = OmegaConf.load(config_path)
    
    # Create model
    model = SequenceLightningModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    model.to(device)
    
    # Get dataset parameters
    obs_keys = config.dataset.get("obs_keys", ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"])
    seq_len = config.dataset.get("seq_len", 10)
    hdf5_path = config.dataset.get("data_path") or config.dataset.get("hdf5_path")
    normalize = config.dataset.get("normalize", True)
    
    print(f"\nModel configuration:")
    print(f"  Observation keys: {obs_keys}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Normalization: {normalize}")
    
    # Create robosuite environment
    # Use same robot/gripper as in dataset (default controller configs)
    
    env = suite.make(
        env_name="Lift",
        robots="Panda",  # Same as robomimic dataset
        has_renderer=render and not save_video,
        has_offscreen_renderer=save_video,
        use_camera_obs=save_video,
        camera_names=camera_name if save_video else None,
        camera_heights=480 if save_video else None,
        camera_widths=640 if save_video else None,
        control_freq=20,  # Same as dataset (20Hz)
        horizon=max_steps,
        reward_shaping=True,
    )
    
    # Create evaluator
    evaluator = RolloutEvaluator(
        model=model,
        env=env,
        obs_keys=obs_keys,
        seq_len=seq_len,
        device=device,
    )
    
    # Set normalization if enabled
    if normalize and hdf5_path:
        evaluator.set_normalization_stats(hdf5_path)
    
    # Video writer
    video_writer = None
    if save_video:
        import imageio
        if video_path is None:
            video_path = Path(checkpoint_path).parent / "rollout_eval.mp4"
        video_writer = imageio.get_writer(video_path, fps=20)
        print(f"\nSaving video to: {video_path}")
    
    # Run rollouts
    print(f"\nRunning {num_rollouts} evaluation rollouts...")
    successes = []
    rewards = []
    lengths = []
    
    for i in range(num_rollouts):
        success, reward, length = evaluator.rollout(
            max_steps=max_steps,
            render=render,
            video_writer=video_writer if (save_video and i == 0) else None,  # Only save first rollout
        )
        
        successes.append(success)
        rewards.append(reward)
        lengths.append(length)
        
        if (i + 1) % 10 == 0:
            print(f"  Rollout {i+1}/{num_rollouts}: Success={success}, Reward={reward:.3f}, Length={length}")
    
    if save_video and video_writer is not None:
        video_writer.close()
        print(f"Video saved to: {video_path}")
    
    # Print results
    success_rate = np.mean(successes)
    avg_reward = np.mean(rewards)
    avg_length = np.mean(lengths)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({num_rollouts} rollouts)")
    print(f"{'='*60}")
    print(f"Success Rate: {success_rate*100:.1f}% ({np.sum(successes)}/{num_rollouts})")
    print(f"Average Reward: {avg_reward:.3f} ± {np.std(rewards):.3f}")
    print(f"Average Length: {avg_length:.1f} ± {np.std(lengths):.1f}")
    print(f"{'='*60}\n")
    
    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "successes": successes,
        "rewards": rewards,
        "lengths": lengths,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LTC/NCP model on robosuite Lift task")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (auto-detected if not specified)")
    parser.add_argument("--num_rollouts", type=int, default=50, help="Number of evaluation rollouts")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per rollout")
    parser.add_argument("--render", action="store_true", help="Render episodes (requires display)")
    parser.add_argument("--save_video", action="store_true", help="Save video of first rollout")
    parser.add_argument("--video_path", type=str, default=None, help="Path to save video")
    parser.add_argument("--camera", type=str, default="agentview", help="Camera name for video")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on")
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        render=args.render,
        save_video=args.save_video,
        video_path=args.video_path,
        camera_name=args.camera,
        device=args.device,
    )


if __name__ == "__main__":
    main()
