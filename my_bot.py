from __future__ import annotations

import argparse
import numpy as np

from game_client import RoomBot
from drive2win import nn as nn_mod
from drive2win.normalize import normalize_states, clip_action


SERVER_URL = "https://ml.ferit.tech"


class TournamentBot:
    def __init__(self, weights_path: str):
        print(f"Loading model: {weights_path}")
        self.w = nn_mod.load(weights_path)

    def make_features(self, obs):
        nav = obs["navigation"]

        speed = float(obs.get("speed", 0.0))
        heading_error = float(nav.get("heading_error", 0.0))
        distance = float(nav.get("distance", 0.0))

        rays = obs.get("rays", [50.0] * 8)
        rays = np.array(rays, dtype=np.float32)

        if rays.shape[0] != 8:
            fixed = np.ones(8, dtype=np.float32) * 50.0
            fixed[: min(8, len(rays))] = rays[: min(8, len(rays))]
            rays = fixed

        ground_friction = float(obs.get("ground_friction", 1.0))

        features = np.concatenate(
            [
                np.array([speed, heading_error, distance], dtype=np.float32),
                rays.astype(np.float32),
                np.array([ground_friction], dtype=np.float32),
            ]
        )

        return features.astype(np.float32)

    def __call__(self, obs):
        try:
            raw_features = self.make_features(obs)

            # Same normalization style as training.
            x = normalize_states(raw_features[None, :])[0]

            throttle, steering = nn_mod.forward(x, self.w)
            throttle, steering = clip_action((throttle, steering))

            return float(throttle), float(steering)

        except Exception as e:
            print("Controller error:", e)

            # Safe fallback: simple checkpoint steering.
            nav = obs.get("navigation", {})
            heading_error = float(nav.get("heading_error", 0.0))
            throttle = 0.45
            steering = np.clip(heading_error * 0.7, -1.0, 1.0)
            return float(throttle), float(steering)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True, help="Tournament room name")
    ap.add_argument("--name", required=True, help="Your bot/name")
    ap.add_argument(
        "--weights",
        default="nav_v3.npz",
        help="Model weights file, for example nav_v3.npz",
    )
    args = ap.parse_args()

    controller = TournamentBot(args.weights)

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