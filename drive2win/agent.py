"""Optional smoother/safety policy for benchmarking.

Use it with:
    python 03_benchmark.py --tag v2-recovery --module drive2win.agent \
        --seeds 42 7 99 --data data_v2-recovery.npz

This file wraps the trained MLP with:
  - action smoothing to reduce jitter
  - a small stuck/wall recovery fallback
"""
from __future__ import annotations
import numpy as np

from . import nn as nn_mod
from .normalize import sensors_to_input, clip_action


def _as_array_action(action):
    """Convert action from tuple/list/np array into np.float32 array."""
    throttle, steering = clip_action(action)
    return np.array([throttle, steering], dtype=np.float32)


class SmoothSafetyPolicy:
    def __init__(self, weights_path: str, alpha: float = 0.70):
        self.w = nn_mod.load(weights_path)
        self.alpha = float(alpha)
        self.prev = np.zeros(2, dtype=np.float32)
        self.stuck_steps = 0

    def __call__(self, state):
        sensors = state.get("sensors", {})

        x = sensors_to_input(sensors)

        # Neural network action
        raw = _as_array_action(nn_mod.forward(x, self.w))

        # Smooth action to reduce jitter
        action = self.alpha * self.prev + (1.0 - self.alpha) * raw
        action = _as_array_action(action)
        self.prev = action

        # Safety fallback.
        # These .get() calls are defensive because sensor names may vary.
        speed = float(sensors.get("speed", sensors.get("velocity", 1.0)))
        front = float(sensors.get("ray_front", sensors.get("front_ray", 1.0)))
        front_left = float(sensors.get("ray_front_left", sensors.get("front_left_ray", 1.0)))
        front_right = float(sensors.get("ray_front_right", sensors.get("front_right_ray", 1.0)))

        near_wall = min(front, front_left, front_right) < 0.20
        slow = abs(speed) < 0.20

        if near_wall and slow:
            self.stuck_steps += 1
        else:
            self.stuck_steps = max(0, self.stuck_steps - 1)

        if self.stuck_steps >= 8:
            # Reverse and turn toward the side with more free space.
            steer = 0.8 if front_left < front_right else -0.8
            action = np.array([-0.65, steer], dtype=np.float32)
            self.prev = action

        throttle, steering = clip_action(action)
        return throttle, steering


def make_policy(weights_path: str):
    return SmoothSafetyPolicy(weights_path)