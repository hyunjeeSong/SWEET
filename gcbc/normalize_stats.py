# python gcbc/normalize_stats.py \
#   --action-dir data/robomimic_can_manual_dataset/action_H16 \
#   --save-path gcbc/robomimic_can_manual_stats.npy

import os
import glob
import argparse
import numpy as np


def compute_stats(action_dir, save_path):
    files = sorted(glob.glob(os.path.join(action_dir, "*.npy")))
    print(f"Loading {len(files)} files from {action_dir} ...")

    if len(files) == 0:
        raise RuntimeError(f"No .npy action files found in: {action_dir}")

    all_actions = []
    bad_files = []

    for fpath in files:
        try:
            act = np.load(fpath)
            if act.shape != (16, 7):
                bad_files.append((fpath, act.shape))
                continue
            all_actions.append(act)
        except Exception as e:
            bad_files.append((fpath, str(e)))

    if len(bad_files) > 0:
        print("=" * 80)
        print(f"Warning: found {len(bad_files)} bad action files. Showing first 20:")
        for item in bad_files[:20]:
            print(item)
        print("=" * 80)

    if len(all_actions) == 0:
        raise RuntimeError("No valid action arrays found after filtering.")

    all_actions = np.concatenate(all_actions, axis=0)  # (N*16, 7)

    min_val = np.min(all_actions, axis=0)
    max_val = np.max(all_actions, axis=0)

    diff = max_val - min_val
    max_val[diff < 1e-6] += 1e-3
    min_val[diff < 1e-6] -= 1e-3

    stats = {
        "min": min_val,
        "max": max_val,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, stats)

    print("Stats computed.")
    print("Min:", min_val)
    print("Max:", max_val)
    print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-dir", required=True, help="Directory containing action_H16 .npy files")
    parser.add_argument("--save-path", required=True, help="Path to save stats .npy")
    args = parser.parse_args()

    compute_stats(args.action_dir, args.save_path)


if __name__ == "__main__":
    main()
