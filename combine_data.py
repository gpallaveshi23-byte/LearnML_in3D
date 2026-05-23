import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output combined dataset name")
    ap.add_argument("files", nargs="+", help="Input .npz data files")
    args = ap.parse_args()

    all_states = []
    all_actions = []
    all_positions = []
    seeds = []

    for file in args.files:
        print(f"Loading {file}")
        d = np.load(file, allow_pickle=False)

        states = d["states"]
        actions = d["actions"]

        print(f"  states : {states.shape}")
        print(f"  actions: {actions.shape}")

        all_states.append(states)
        all_actions.append(actions)

        if "positions" in d.files:
            all_positions.append(d["positions"])

        if "seed" in d.files:
            seeds.append(d["seed"])

    states_out = np.concatenate(all_states, axis=0)
    actions_out = np.concatenate(all_actions, axis=0)

    print()
    print("Combined:")
    print("  states :", states_out.shape)
    print("  actions:", actions_out.shape)

    save_dict = {
        "states": states_out,
        "actions": actions_out,
    }

    if all_positions:
        save_dict["positions"] = np.concatenate(all_positions, axis=0)

    if seeds:
        save_dict["seed"] = np.array(seeds)

    np.savez(args.out, **save_dict)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()