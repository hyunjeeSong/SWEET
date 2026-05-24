# python gcbc/process_manual_task_data.py \
#   --task lift \
#   --h5-path data/robomimic_image/lift/ph/image_v15_1280.hdf5 \
#   --meta-dir data/robomimic/robomimic_result/lift/meta \
#   --gt-final-dir data/robomimic/robomimic_result/lift/final \
#   --inf-final-dir data/robomimic_inf/robomimic_inf_flux \
#   --output-root data/robomimic_lift_manual_dataset

import os
import glob
import json
import argparse
import shutil
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


SEQ_LEN = 16
STRIDE = 8


def pad_actions(actions, target_len=SEQ_LEN):
    if len(actions) == 0:
        raise ValueError("Cannot pad empty action sequence.")
    pad_len = target_len - len(actions)
    if pad_len > 0:
        last_action = actions[-1:]
        padding = np.repeat(last_action, pad_len, axis=0)
        return np.concatenate([actions, padding], axis=0)
    return actions


def ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def normalize_demo_id(video_id, task):
    """
    Convert possible task-prefixed ids such as can_demo_0 back to demo_0.
    """
    if video_id.startswith(f"{task}_"):
        return video_id[len(task) + 1:]
    return video_id


def find_goal_image(goal_dir, task, demo_id, step_key):
    """
    Supports:
      demo_0_step1.png
      can_demo_0_step1.png
      demo_0_step1_*.png
      can_demo_0_step1_*.png
    """
    candidates = [
        os.path.join(goal_dir, f"{demo_id}_{step_key}.png"),
        os.path.join(goal_dir, f"{task}_{demo_id}_{step_key}.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    patterns = [
        os.path.join(goal_dir, f"{demo_id}_{step_key}_*.png"),
        os.path.join(goal_dir, f"{task}_{demo_id}_{step_key}_*.png"),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]

    return None


def save_one_sample(
    prefix_name,
    chunk_start,
    chunk_actions,
    all_imgs,
    gt_goal_path,
    inf_goal_path,
    out_action,
    out_start_img,
    out_goal_gt,
    out_goal_inf,
    out_action_mix,
    out_start_mix,
    out_goal_mix,
):
    # Base GT / INF dataset
    np.save(os.path.join(out_action, f"{prefix_name}.npy"), chunk_actions)

    img_np = all_imgs[chunk_start]
    Image.fromarray(img_np).save(os.path.join(out_start_img, f"{prefix_name}.png"))

    shutil.copy2(gt_goal_path, os.path.join(out_goal_gt, f"{prefix_name}.png"))

    has_inf = inf_goal_path is not None and os.path.exists(inf_goal_path)
    if has_inf:
        shutil.copy2(inf_goal_path, os.path.join(out_goal_inf, f"{prefix_name}.png"))

    # Mixed dataset: one GT copy and one INF copy with the same action label
    mix_gt_name = f"{prefix_name}_gt"
    np.save(os.path.join(out_action_mix, f"{mix_gt_name}.npy"), chunk_actions)
    Image.fromarray(img_np).save(os.path.join(out_start_mix, f"{mix_gt_name}.png"))
    shutil.copy2(gt_goal_path, os.path.join(out_goal_mix, f"{mix_gt_name}.png"))

    if has_inf:
        mix_inf_name = f"{prefix_name}_inf"
        np.save(os.path.join(out_action_mix, f"{mix_inf_name}.npy"), chunk_actions)
        Image.fromarray(img_np).save(os.path.join(out_start_mix, f"{mix_inf_name}.png"))
        shutil.copy2(inf_goal_path, os.path.join(out_goal_mix, f"{mix_inf_name}.png"))


def get_chunk_rule(task, step_key):
    """
    Rule A:
      Every 8 steps take a full 16-step chunk,
      and additionally save the final last-16 chunk.

    Rule B:
      Every 8 steps take a full 16-step chunk,
      and pad the final residual chunk if it is shorter than 16.

    lift:
      step1 -> Rule A
      step2 -> Rule B

    can / square:
      step1, step2 -> Rule A
      step3        -> Rule B
    """
    if task == "lift":
        if step_key == "step1":
            return "A"
        if step_key == "step2":
            return "B"
        return None

    if task in ["can", "square"]:
        if step_key in ["step1", "step2"]:
            return "A"
        if step_key == "step3":
            return "B"
        return None

    return None


def process_step_chunks(
    task,
    demo_id,
    step_key,
    start_i,
    end_i,
    all_actions,
    all_imgs,
    gt_goal_path,
    inf_goal_path,
    output_dirs,
):
    out_action = output_dirs["action"]
    out_start_img = output_dirs["start"]
    out_goal_gt = output_dirs["goal_gt"]
    out_goal_inf = output_dirs["goal_inf"]
    out_action_mix = output_dirs["action_mix"]
    out_start_mix = output_dirs["start_mix"]
    out_goal_mix = output_dirs["goal_mix"]

    if end_i < start_i:
        return 0

    rule = get_chunk_rule(task, step_key)
    if rule is None:
        return 0

    n_saved = 0
    saved_starts = set()

    def save_chunk(chunk_start, chunk_actions, suffix=""):
        nonlocal n_saved
        prefix_name = f"{demo_id}_{step_key}_{chunk_start}{suffix}"
        save_one_sample(
            prefix_name=prefix_name,
            chunk_start=chunk_start,
            chunk_actions=chunk_actions,
            all_imgs=all_imgs,
            gt_goal_path=gt_goal_path,
            inf_goal_path=inf_goal_path,
            out_action=out_action,
            out_start_img=out_start_img,
            out_goal_gt=out_goal_gt,
            out_goal_inf=out_goal_inf,
            out_action_mix=out_action_mix,
            out_start_mix=out_start_mix,
            out_goal_mix=out_goal_mix,
        )
        n_saved += 1

    # Rule A: full 16-step chunks + final last16
    if rule == "A":
        curr = start_i

        while curr + SEQ_LEN - 1 <= end_i:
            actions = all_actions[curr: curr + SEQ_LEN]
            if len(actions) == SEQ_LEN:
                save_chunk(curr, actions)
                saved_starts.add(curr)
            curr += STRIDE

        last_start = end_i - SEQ_LEN + 1
        if last_start >= start_i and last_start not in saved_starts:
            actions = all_actions[last_start: end_i + 1]
            if len(actions) == SEQ_LEN:
                save_chunk(last_start, actions, suffix="_last16")
                saved_starts.add(last_start)

    # Rule B: full 16-step chunks + padded final residual
    elif rule == "B":
        curr = start_i

        while curr + SEQ_LEN - 1 <= end_i:
            actions = all_actions[curr: curr + SEQ_LEN]
            if len(actions) == SEQ_LEN:
                save_chunk(curr, actions)
                saved_starts.add(curr)
            curr += STRIDE

        # Padding final residual chunk if it is not already exactly covered
        if curr <= end_i:
            actions = all_actions[curr: end_i + 1]
            actions = pad_actions(actions, SEQ_LEN)
            save_chunk(curr, actions, suffix="_padded")
            saved_starts.add(curr)

        # If the whole segment is shorter than 16, while loop will not run and this branch handles it
        if len(saved_starts) == 0:
            actions = all_actions[start_i: end_i + 1]
            actions = pad_actions(actions, SEQ_LEN)
            save_chunk(start_i, actions, suffix="_padded")
            saved_starts.add(start_i)

    return n_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["can", "square", "lift"])
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--meta-dir", required=True)
    parser.add_argument("--gt-final-dir", required=True)
    parser.add_argument("--inf-final-dir", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = args.output_root

    output_dirs = {
        "action": os.path.join(output_root, "action_H16"),
        "start": os.path.join(output_root, "images_first_H16"),
        "goal_gt": os.path.join(output_root, "images_goal_gt_H16"),
        "goal_inf": os.path.join(output_root, "images_goal_inf_H16"),

        "action_mix": os.path.join(output_root, "action_mix_H16"),
        "start_mix": os.path.join(output_root, "images_first_mix_H16"),
        "goal_mix": os.path.join(output_root, "images_goal_mix_H16"),
    }
    ensure_dirs(output_dirs.values())

    meta_files = sorted([f for f in os.listdir(args.meta_dir) if f.endswith(".json")])

    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"H5: {args.h5_path}")
    print(f"Meta dir: {args.meta_dir}")
    print(f"GT final dir: {args.gt_final_dir}")
    print(f"INF final dir: {args.inf_final_dir}")
    print(f"Output root: {output_root}")
    print(f"Found meta files: {len(meta_files)}")
    print("Chunking rule:")
    if args.task == "lift":
        print("  lift step1 -> full chunks + final last16")
        print("  lift step2 -> full chunks + padded final residual")
    else:
        print("  can/square step1, step2 -> full chunks + final last16")
        print("  can/square step3        -> full chunks + padded final residual")
    print("=" * 80)

    total_saved = 0
    missing_h5 = 0
    missing_gt = 0
    missing_inf = 0
    bad_meta = 0
    skipped_unknown_step = 0

    with h5py.File(args.h5_path, "r") as f_h5:
        for meta_file in tqdm(meta_files, desc=f"Processing {args.task}"):
            meta_path = os.path.join(args.meta_dir, meta_file)

            try:
                with open(meta_path, "r") as jf:
                    meta = json.load(jf)
            except Exception:
                bad_meta += 1
                continue

            demo_id = meta.get("video_id")
            step_key = meta.get("step_name")

            if not demo_id or not step_key or "A" not in meta or "B" not in meta:
                bad_meta += 1
                continue

            demo_id = normalize_demo_id(demo_id, args.task)

            if get_chunk_rule(args.task, step_key) is None:
                skipped_unknown_step += 1
                continue

            try:
                start_i = int(meta["A"]["frame_index"])
                end_i = int(meta["B"]["frame_index"])
            except Exception:
                bad_meta += 1
                continue

            if f"data/{demo_id}" not in f_h5:
                missing_h5 += 1
                continue

            gt_goal_path = find_goal_image(args.gt_final_dir, args.task, demo_id, step_key)
            inf_goal_path = find_goal_image(args.inf_final_dir, args.task, demo_id, step_key)

            if gt_goal_path is None:
                missing_gt += 1
                continue

            if inf_goal_path is None:
                missing_inf += 1

            demo_grp = f_h5[f"data/{demo_id}"]
            all_actions = demo_grp["actions"][:]
            all_imgs = demo_grp["obs/agentview_image"]

            # Clamp index range defensively
            max_action_idx = len(all_actions) - 1
            max_img_idx = len(all_imgs) - 1
            max_valid_idx = min(max_action_idx, max_img_idx)

            if start_i < 0:
                start_i = 0
            if end_i > max_valid_idx:
                end_i = max_valid_idx
            if end_i < start_i:
                bad_meta += 1
                continue

            try:
                n = process_step_chunks(
                    task=args.task,
                    demo_id=demo_id,
                    step_key=step_key,
                    start_i=start_i,
                    end_i=end_i,
                    all_actions=all_actions,
                    all_imgs=all_imgs,
                    gt_goal_path=gt_goal_path,
                    inf_goal_path=inf_goal_path,
                    output_dirs=output_dirs,
                )
                total_saved += n
            except Exception:
                bad_meta += 1
                continue

    print("=" * 80)
    print("Finished.")
    print(f"Total base chunks saved: {total_saved}")
    print(f"Missing H5 demos: {missing_h5}")
    print(f"Missing GT goals: {missing_gt}")
    print(f"Missing INF goals: {missing_inf}")
    print(f"Bad meta files: {bad_meta}")
    print(f"Skipped unknown-step meta files: {skipped_unknown_step}")
    print(f"Output root: {output_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
