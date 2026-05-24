import numpy as np


def angle_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def simulate_run(seed=0, steps=1000):
    rng = np.random.default_rng(seed)

    # Checkpoints for a simple driving track
    t = np.linspace(0, 1, 8)
    checkpoints = np.column_stack([
        1000 * t,
        180 * np.sin(2.2 * np.pi * t)
    ])

    pos = checkpoints[0].astype(float) + np.array([0.0, -20.0])
    heading = 0.0
    speed = 0.0
    target_idx = 1

    states = []
    actions = []

    for step in range(steps):
        target = checkpoints[target_idx]
        vec = target - pos
        dist = np.linalg.norm(vec)

        target_angle = np.arctan2(vec[1], vec[0])
        angle_error = angle_wrap(target_angle - heading)

        steering = np.clip(angle_error / 0.8, -1, 1)
        throttle = np.clip(1.0 - abs(angle_error) / 1.4, 0.25, 1.0)

        steering = np.clip(steering + rng.normal(0, 0.03), -1, 1)
        throttle = np.clip(throttle + rng.normal(0, 0.03), 0, 1)

        heading = angle_wrap(heading + steering * 0.045)
        speed = np.clip(speed + 0.18 * throttle - 0.04 * speed, 0, 8.5)

        pos = pos + np.array([
            np.cos(heading),
            np.sin(heading)
        ]) * speed

        if dist < 45 and target_idx < len(checkpoints) - 1:
            target_idx += 1

        next_checkpoint = checkpoints[target_idx]
        next_vec = next_checkpoint - pos
        next_dist = np.linalg.norm(next_vec)

        next_angle = np.arctan2(next_vec[1], next_vec[0])
        next_angle_error = angle_wrap(next_angle - heading)

        state = [
            pos[0] / 1000,
            pos[1] / 300,
            np.cos(heading),
            np.sin(heading),
            speed / 10,
            next_vec[0] / 1000,
            next_vec[1] / 300,
            next_dist / 1000,
            np.cos(next_angle_error),
            np.sin(next_angle_error),
            target_idx / (len(checkpoints) - 1),
            1.0
        ]

        action = [
            steering,
            throttle
        ]

        states.append(state)
        actions.append(action)

    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)


def main():
    all_states = []
    all_actions = []

    number_of_runs = 35
    steps_per_run = 1000

    for seed in range(number_of_runs):
        states, actions = simulate_run(seed=seed, steps=steps_per_run)
        all_states.append(states)
        all_actions.append(actions)

    states = np.vstack(all_states)
    actions = np.vstack(all_actions)

    print("Dataset created:")
    print("states :", states.shape)
    print("actions:", actions.shape)

    # Save everything into ONE .npz file
    np.savez("point.npz", states=states, actions=actions)

    print("Saved: point.npz")


if __name__ == "__main__":
    main()