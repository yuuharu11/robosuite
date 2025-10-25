"""
Simplified rollout evaluator for robosuite Lift task.
Random policy version (baseline).
"""

import robosuite as suite
import numpy as np


def rollout(env, max_steps=200):
    obs = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        low, high = env.action_spec
        action = np.random.uniform(low, high)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if env.has_renderer:
            env.render()

        if done:
            break

    return total_reward, step + 1


def main():
    # Basic Lift task environment (same robot as training)
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    print("✅ Starting simple rollout (random actions)")
    reward, length = rollout(env)
    print(f"Rollout finished. Reward = {reward:.3f}, Episode Length = {length}")

    env.close()


if __name__ == "__main__":
    main()
