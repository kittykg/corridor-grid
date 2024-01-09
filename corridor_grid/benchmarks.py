import argparse
import time

import gymnasium as gym

import corridor_grid


def benchmark(env_id: str, num_resets: int, num_frames: int):
    env = gym.make(env_id, render_mode="rgb_array")
    # Benchmark env.reset
    print("Benchmarking env.reset()...")
    t0 = time.time()
    for _ in range(num_resets):
        env.reset()
    t1 = time.time()
    dt = t1 - t0
    reset_time = (1000 * dt) / num_resets

    # Benchmark rendering
    print("Benchmarking env.render()...")
    t0 = time.time()
    for _ in range(num_frames):
        env.render()
    t1 = time.time()
    dt = t1 - t0
    fps = num_frames / dt

    # Benchmark rendering in agent view
    env.reset()
    print("Benchmarking env.step(0)...")
    t0 = time.time()
    for _ in range(num_frames):
        env.step(0)
    t1 = time.time()
    dt = t1 - t0
    agent_view_fps = num_frames / dt

    print(f"Env ID        : {env_id}")
    print(f"Env reset time: {reset_time:.1f} ms")
    print(f"Rendering FPS : {fps:.0f}")
    print(f"Agent view FPS: {agent_view_fps:.0f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        dest="env_id",
        help="gym environment to load",
        default="CG-SC-v0",
    )
    parser.add_argument(
        "--num-resets",
        type=int,
        help="number of times to reset the environment for benchmarking",
        default=200,
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="number of frames to test rendering for",
        default=100,
    )

    args = parser.parse_args()
    benchmark(args.env_id, args.num_resets, args.num_frames)
