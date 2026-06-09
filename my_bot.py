from __future__ import annotations

import argparse
import numpy as np

from game_client import RoomBot
from drive2win.normalize import clip_action


SERVER_URL = "https://ml.ferit.tech"


# Change this only if the bot turns the wrong way.
# Use 1.0 first. If it circles away from checkpoints, change to -1.0.
STEERING_SIGN = 1.0


def relu(x):
    return np.maximum(0, x)


def forward_mlp(x, w):
    z1 = x @ w["W1"] + w["b1"]
    a1 = relu(z1)

    z2 = a1 @ w["W2"] + w["b2"]
    a2 = relu(z2)

    z3 = a2 @ w["W3"] + w["b3"]
    y = np.tanh(z3)

    return y


class TournamentAgent:
    def __init__(self, weights_path: str):
        print(f"Loading model: {weights_path}")

        z = np.load(weights_path, allow_pickle=True)

        self.w = {
            "W1": z["W1"].astype(np.float32),
            "b1": z["b1"].astype(np.float32),
            "W2": z["W2"].astype(np.float32),
            "b2": z["b2"].astype(np.float32),
            "W3": z["W3"].astype(np.float32),
            "b3": z["b3"].astype(np.float32),
        }

        self.normalizer = None
        if "normalizer" in z.files:
            try:
                self.normalizer = z["normalizer"].item()
                print("Loaded saved normalizer.")
            except Exception as e:
                print("Could not load normalizer:", e)

        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self.position_history = []
        self.stuck_steps = 0
        self.recovery_steps = 0
        self.recovery_steer = 0.7

        print("Model shapes:")
        for k, v in self.w.items():
            print(f"  {k}: {v.shape}")

    def make_features(self, obs):
        nav = obs.get("navigation", {})

        speed = float(obs.get("speed", 0.0))
        heading_error = float(nav.get("heading_error", 0.0))
        distance = float(nav.get("distance", 0.0))

        rays = np.array(obs.get("rays", [50.0] * 8), dtype=np.float32)
        if len(rays) != 8:
            fixed = np.ones(8, dtype=np.float32) * 50.0
            n = min(8, len(rays))
            fixed[:n] = rays[:n]
            rays = fixed

        ground_friction = float(obs.get("ground_friction", 1.0))

        features = np.concatenate(
            [
                np.array([speed, heading_error, distance], dtype=np.float32),
                rays,
                np.array([ground_friction], dtype=np.float32),
            ]
        )

        return features.astype(np.float32)

    def normalize_features(self, features):
        if self.normalizer is not None:
            mean = np.array(self.normalizer["mean"], dtype=np.float32)
            std = np.array(self.normalizer["std"], dtype=np.float32)
            return (features - mean) / (std + 1e-8)

        return features

    def checkpoint_fallback(self, obs):
        nav = obs.get("navigation", {})
        heading_error = float(nav.get("heading_error", 0.0))

        steering = STEERING_SIGN * np.clip(heading_error * 0.45, -0.7, 0.7)

        turn = abs(heading_error)
        if turn > 1.2:
            throttle = 0.25
        elif turn > 0.7:
            throttle = 0.40
        else:
            throttle = 0.60

        return np.array([throttle, steering], dtype=np.float32)

    def get_position(self, obs):
        pos = obs.get("position", {})
        return np.array(
            [
                float(pos.get("x", 0.0)),
                float(pos.get("z", 0.0)),
            ],
            dtype=np.float32,
        )

    def update_stuck_detection(self, obs):
        pos = self.get_position(obs)
        self.position_history.append(pos)

        if len(self.position_history) > 25:
            self.position_history.pop(0)

        moved_little = False
        if len(self.position_history) >= 25:
            dist = np.linalg.norm(self.position_history[-1] - self.position_history[0])
            moved_little = dist < 0.25

        speed = float(obs.get("speed", 0.0))
        rays = np.array(obs.get("rays", [50.0] * 8), dtype=np.float32)

        front = rays[0] if len(rays) > 0 else 50.0
        front_left = rays[1] if len(rays) > 1 else 50.0
        front_right = rays[7] if len(rays) > 7 else 50.0

        near_wall = min(front, front_left, front_right) < 2.0
        very_slow = abs(speed) < 0.20

        if moved_little or (near_wall and very_slow):
            self.stuck_steps += 1
        else:
            self.stuck_steps = max(0, self.stuck_steps - 1)

        return self.stuck_steps >= 14

    def choose_recovery_steer(self, obs):
        rays = np.array(obs.get("rays", [50.0] * 8), dtype=np.float32)

        left_space = rays[1] if len(rays) > 1 else 50.0
        right_space = rays[7] if len(rays) > 7 else 50.0

        if right_space > left_space:
            return 0.7
        return -0.7

    def __call__(self, obs):
        try:
            # Recovery mode: reverse and turn briefly.
            if self.recovery_steps > 0:
                self.recovery_steps -= 1
                action = np.array([-0.55, self.recovery_steer], dtype=np.float32)
                self.prev_action = action
                throttle, steering = clip_action(action)
                return float(throttle), float(steering)

            # Neural network action.
            features = self.make_features(obs)
            x = self.normalize_features(features)
            nn_action = forward_mlp(x, self.w)

            throttle, steering = clip_action(nn_action)
            action = np.array([float(throttle), float(steering)], dtype=np.float32)

            # Apply steering sign safely.
            action[1] = STEERING_SIGN * action[1]

            # Reduce steering strength to avoid circles.
            action[1] *= 0.55

            # Light smoothing.
            alpha = 0.30
            action = alpha * self.prev_action + (1.0 - alpha) * action

            # Limit steering.
            action[1] = np.clip(action[1], -0.75, 0.75)

            # If the model is weak or confused, use checkpoint fallback.
            if abs(action[0]) < 0.08:
                action = self.checkpoint_fallback(obs)

            # If turning too hard while barely moving, use fallback.
            if abs(action[1]) > 0.65 and abs(action[0]) < 0.20:
                action = self.checkpoint_fallback(obs)

            # Stuck recovery.
            if self.update_stuck_detection(obs):
                self.stuck_steps = 0
                self.recovery_steps = 16
                self.recovery_steer = self.choose_recovery_steer(obs)
                action = np.array([-0.55, self.recovery_steer], dtype=np.float32)

            self.prev_action = action.astype(np.float32)

            throttle, steering = clip_action(action)
            return float(throttle), float(steering)

        except Exception as e:
            print("Controller error:", e)
            action = self.checkpoint_fallback(obs)
            throttle, steering = clip_action(action)
            return float(throttle), float(steering)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True, help="Tournament room name")
    ap.add_argument("--name", required=True, help="Your bot name")
    ap.add_argument(
        "--weights",
        default="nav_final.npz",
        help="Model weights file, for example nav_final.npz",
    )
    args = ap.parse_args()

    controller = TournamentAgent(args.weights)

    bot = RoomBot(
        SERVER_URL,
        room=args.room,
        name=args.name,
    )

    standings = bot.run(controller, hz=20.0)

    print("Final standings:")
    print(standings)


if __name__ == "__main__":
    main()