"""Step 1 — Collect data with obstacles turned off.

Run:
    py 01_collect.py --tag v6-no-obstacles --seed 42

Output:
    data_<tag>.npz

Example:
    data_v6-no-obstacles.npz
"""
from __future__ import annotations

import argparse
import threading
import time
import numpy as np

from game_client import GameClient


SERVER_URL = "https://ml.ferit.tech"
API_KEY = "None"  # paste yours if the server requires it


# No obstacle phase here because we are trying to disable obstacles.
PHASES = [
    (
        "Smooth checkpoint driving",
        180,
        "Drive slowly and smoothly toward checkpoints. Try to complete as many checkpoints as possible.",
    ),
    (
        "Tight turns",
        120,
        "Slow before corners, then steer smoothly through them. Do not crash into walls.",
    ),
    (
        "Bad terrain",
        60,
        "Drive deliberate lines on ice, mud, and sand. Avoid sudden steering.",
    ),
    (
        "Small wall recovery",
        30,
        "Touch walls only a little, then reverse, turn away, and continue driving normally.",
    ),
]


def _poll_positions(client: GameClient, stop_evt: threading.Event, out: list, hz: float = 5.0):
    """Background thread: poll position at low Hz so we can plot the path later."""
    interval = 1.0 / hz

    while not stop_evt.is_set():
        try:
            st = client.get_latest_state()
            pos = st.get("position") if st else None

            if pos and "x" in pos and "z" in pos:
                out.append((time.time(), pos["x"], pos["z"]))

        except Exception:
            pass

        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tag",
        default="v6-no-obstacles",
        help="Suffix for output file. Example: --tag v6-no-obstacles saves data_v6-no-obstacles.npz",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Map seed. Keep fixed while comparing versions.",
    )
    args = ap.parse_args()

    client = GameClient(SERVER_URL, API_KEY)

    # Obstacle-disabling config.
    # The server may accept only one of these keys.
    # Extra unknown keys are usually ignored.
    session_config = {
        "seed": args.seed,
        "wind_enabled": False,

        # Try to disable obstacles:
        "obstacles_enabled": False,
        "obstacles": False,
        "num_obstacles": 0,
        "obstacle_count": 0,
    }

    session = client.create_session(
        mode="time_trial",
        player_name=f"d2w_collector_{args.tag}",
        config=session_config,
    )

    print("Session config:")
    print(session_config)
    print()

    print("Open this URL in a NEW TAB and click into it so WASD reach the game:")
    print(" ", session.get("browser_url"))
    print()
    input("Press Enter once the browser tab has focus and you can see the bot. ")

    client.connect_ws()
    time.sleep(0.5)

    positions = []
    stop_evt = threading.Event()

    t = threading.Thread(
        target=_poll_positions,
        args=(client, stop_evt, positions),
        daemon=True,
    )
    t.start()

    client.start_recording(sample_rate=20)

    for i, (name, seconds, hint) in enumerate(PHASES, 1):
        print(f"\n--- Phase {i}/{len(PHASES)} — {name} ({seconds}s) ---")
        print(f"  {hint}")
        print("  Switch to the browser tab and drive now.")

        remaining = seconds

        while remaining > 0:
            print(f"  ... {remaining}s remaining")
            sleep_time = min(10, remaining)
            time.sleep(sleep_time)
            remaining -= sleep_time

    stop_evt.set()

    info = client.stop_recording()
    print(f"\nStopped. Samples on the server: {info.get('sample_count', '?')}")

    states_raw, actions = client.get_recording_as_arrays()

    print(f"states shape   : {states_raw.shape}   (N, 12)")
    print(f"actions shape  : {actions.shape}      (N, 2)")

    pos_arr = np.array([(p[1], p[2]) for p in positions], dtype=np.float32)
    print(f"positions shape: {pos_arr.shape}     (M, 2)")

    assert states_raw.shape[0] >= 5_000, (
        "Fewer than 5,000 samples. Drive more before saving."
    )

    out = f"data_{args.tag}.npz"

    np.savez(
        out,
        states=states_raw,
        actions=actions,
        positions=pos_arr,
        seed=args.seed,
    )

    print(f"Saved {out}")

    try:
        client.disconnect_ws()
        client.delete_session()
    except Exception:
        pass


if __name__ == "__main__":
    main()