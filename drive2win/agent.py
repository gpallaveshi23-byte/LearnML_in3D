"""Smoother/safety policy for benchmarking.

Use:
    python 03_benchmark.py --tag v4-finish --module drive2win.agent \
        --seeds 42 --data data_v4-finish.npz
"""
from __future__ import annotations
import numpy as np

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action


def _as_array_action(action):
    throttle, steering = clip_action(action)
    return np.array([throttle, steering], dtype=np.float32)


class SmoothSafetyPolicy:
    def __init__(self, weights_path: str, alpha: float = 0.60):
        self.w = nn_mod.load(weights_path)
        self.alpha = float(alpha)
        self.prev = np.zeros(2, dtype=np.float32)

        self.position_history = []
        self.stuck_steps = 0
        self.recovery_steps = 0
        self.recovery_steer = 0.8

    def _get_position(self, state):
        pos = state.get("position", None)
        if isinstance(pos, dict) and "x" in pos and "z" in pos:
            return np.array([float(pos["x"]), float(pos["z"])], dtype=np.float32)
        return None

    def _get_ray(self, sensors, *names, default=1.0):
        for name in names:
            if name in sensors:
                try:
                    return float(sensors[name])
                except Exception:
                    pass
        return default

    def __call__(self, state):
        sensors = state.get("sensors", {})

        # If already in recovery mode, reverse and turn for a short time.
        if self.recovery_steps > 0:
            self.recovery_steps -= 1
            action = np.array([-0.65, self.recovery_steer], dtype=np.float32)
            self.prev = action
            throttle, steering = clip_action(action)
            return throttle, steering

        # Normal neural-network action.
        x = sensors_to_input(sensors)
        raw = _as_array_action(nn_mod.forward(x, self.w))

        # Smooth action.
        action = self.alpha * self.prev + (1.0 - self.alpha) * raw
        action = _as_array_action(action)
        self.prev = action

        # Track movement using position history.
        pos = self._get_position(state)
        if pos is not None:
            self.position_history.append(pos)
            if len(self.position_history) > 20:
                self.position_history.pop(0)

        moved_little = False
        if len(self.position_history) >= 20:
            dist = np.linalg.norm(self.position_history[-1] - self.position_history[0])
            moved_little = dist < 0.25

        # Try several possible ray names.
        front = self._get_ray(
            sensors,
            "ray_front",
            "front_ray",
            "front",
            "distance_front",
            default=1.0,
        )
        front_left = self._get_ray(
            sensors,
            "ray_front_left",
            "front_left_ray",
            "front_left",
            "distance_front_left",
            default=1.0,
        )
        front_right = self._get_ray(
            sensors,
            "ray_front_right",
            "front_right_ray",
            "front_right",
            "distance_front_right",
            default=1.0,
        )

        near_wall = min(front, front_left, front_right) < 0.25

        # If the bot has not moved much or is near a wall, count stuck frames.
        if moved_little or near_wall:
            self.stuck_steps += 1
        else:
            self.stuck_steps = max(0, self.stuck_steps - 1)

        # Start recovery after enough stuck frames.
        if self.stuck_steps >= 12:
            self.stuck_steps = 0
            self.recovery_steps = 18

            # Turn toward the side with more space.
            if front_left < front_right:
                self.recovery_steer = 0.9
            else:
                self.recovery_steer = -0.9

            action = np.array([-0.70, self.recovery_steer], dtype=np.float32)
            self.prev = action
            throttle, steering = clip_action(action)
            return throttle, steering

        throttle, steering = clip_action(action)
        return throttle, steering


def make_policy(weights_path: str):
    return SmoothSafetyPolicy(weights_path)