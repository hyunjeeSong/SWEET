# python batch_inference_benchmark.py \
#   --task can \
#   --ckpt-type mix \
#   --subgoal-type inf \
#   --num-demos 20 \
#   --ckpt-path gcbc/checkpoints_robomimic_can_manual_mix/gcbc_ckpt_100.pth \
#   --output-json gcbc/benchmark_can_mixmodel_ckpt100_infsubgoal.json 

import os
import glob
import json
import csv
import argparse
import time
import numpy as np
import torch
import h5py
import robosuite as suite
from PIL import Image
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
import sys

GCBC_ROOT = "gcbc"
if GCBC_ROOT not in sys.path:
    sys.path.append(GCBC_ROOT)

from gcbc_model import VisionEncoder, DiffusionPolicyGCBC


TASK_CONFIG = {
    "lift": {
        "env_name": "Lift",
        "h5_path": "data/robomimic_image/lift/ph/image_v15_1280.hdf5",
        "stats_path": "gcbc/robomimic_lift_manual_stats.npy",
        # 如果你的 lift meta 不在这个位置，请运行时用 --meta-dir 覆盖
        "meta_dir": "data/robomimic/robomimic_result/lift/meta",
        "gt_subgoal_root": "data/robomimic_lift_manual_dataset/images_goal_gt_H16",
        "inf_subgoal_root": "data/robomimic_lift_manual_dataset/images_goal_inf_H16",
        "ckpt_dirs": {
            "gt": "gcbc/checkpoints_robomimic_lift_manual",
            "inf": "gcbc/checkpoints_robomimic_lift_manual_inf",
            "mix": "gcbc/checkpoints_robomimic_lift_manual_mix",
        },
        "subgoal_sequence": ["step1", "step2"],
        "compare_steps": ["step1"],
        "max_steps_per_subgoal": 48,
        "inf_name_prefix": "",
    },
    "can": {
        "env_name": "PickPlaceCan",
        "h5_path": "data/robomimic_image/can/ph/image_v15_1280.hdf5",
        "stats_path": "gcbc/robomimic_can_manual_stats.npy",
        "meta_dir": "data/robomimic/robomimic_result/can/meta",
        "gt_subgoal_root": "data/robomimic/robomimic_result/can/final",
        "inf_subgoal_root": "data/robomimic_inf/robomimic_inf_flux_6000step_30infstep",
        "ckpt_dirs": {
            "gt": "gcbc/checkpoints_robomimic_can_manual",
            "inf": "gcbc/checkpoints_robomimic_can_manual_inf",
            "mix": "gcbc/checkpoints_robomimic_can_manual_mix",
        },
        "subgoal_sequence": ["step1", "step2", "step3"],
        "compare_steps": ["step1", "step2"],
        "max_steps_per_subgoal": 64,
        "inf_name_prefix": "can_",
    },
    "square": {
        "env_name": "NutAssemblySquare",
        "h5_path": "data/robomimic_image/square/ph/image_v15_1280.hdf5",
        "stats_path": "gcbc/robomimic_square_manual_stats.npy",
        "meta_dir": "data/robomimic/robomimic_result/square/meta",
        "gt_subgoal_root": "data/robomimic/robomimic_result/square/final",
        "inf_subgoal_root": "data/robomimic_inf/robomimic_inf_flux_6000step_30infstep",
        "ckpt_dirs": {
            "gt": "gcbc/checkpoints_robomimic_square_manual",
            "inf": "gcbc/checkpoints_robomimic_square_manual_inf",
            "mix": "gcbc/checkpoints_robomimic_square_manual_mix",
        },
        "subgoal_sequence": ["step1", "step2", "step3"],
        "compare_steps": ["step1", "step2"],
        "max_steps_per_subgoal": 64,
        "inf_name_prefix": "square_",
    },
}


SEED = 42
ACTION_CHUNK_SIZE = 16
EXEC_HORIZON = 16
WARMUP_STEPS = 10
RENDER_H = 720
RENDER_W = 1280


def latest_ckpt(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "gcbc_ckpt_*.pth"))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    def epoch_num(p):
        base = os.path.basename(p)
        return int(base.replace("gcbc_ckpt_", "").replace(".pth", ""))

    return sorted(files, key=epoch_num)[-1]


def get_subgoal_path(task, subgoal_type, demo_id, step_name, cfg):
    if subgoal_type == "gt":
        root = cfg["gt_subgoal_root"]
        direct = os.path.join(root, f"{demo_id}_{step_name}.png")
        if os.path.exists(direct):
            return direct
        matches = sorted(glob.glob(os.path.join(root, f"{demo_id}_{step_name}_*.png")))
        if matches:
            return matches[0]
        return direct

    if subgoal_type == "inf":
        root = cfg["inf_subgoal_root"]
        prefix = cfg.get("inf_name_prefix", "")
        direct = os.path.join(root, f"{prefix}{demo_id}_{step_name}.png")
        if os.path.exists(direct):
            return direct
        matches = sorted(glob.glob(os.path.join(root, f"{prefix}{demo_id}_{step_name}_*.png")))
        if matches:
            return matches[0]
        return direct

    raise ValueError(f"Unknown subgoal_type: {subgoal_type}")


def has_all_subgoals(task, subgoal_type, demo_id, cfg):
    return all(
        os.path.exists(get_subgoal_path(task, subgoal_type, demo_id, step_name, cfg))
        for step_name in cfg["subgoal_sequence"]
    )


def load_meta_frame_indices(meta_dir, demo_id, compare_steps):
    """
    Return B-frame indices for steps to compare.
    For lift: step1 only.
    For can/square: step1 and step2.
    """
    out = {}
    for step_name in compare_steps:
        meta_path = os.path.join(meta_dir, f"{demo_id}_{step_name}.json")
        if not os.path.exists(meta_path):
            return None, f"Missing meta: {meta_path}"

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if "B" not in meta or "frame_index" not in meta["B"]:
            return None, f"Bad meta format: {meta_path}"

        out[step_name] = int(meta["B"]["frame_index"])

    return out, ""


def make_env(env_name):
    controller_config = load_composite_controller_config(robot="Panda")
    arm_key = "right" if "right" in controller_config["body_parts"] else "right_arm"
    arm_config = controller_config["body_parts"][arm_key]
    arm_config.update({
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "control_delta": True,
    })

    env = suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=RENDER_H,
        camera_widths=RENDER_W,
        control_freq=20,
    )
    return env, arm_key


def reset_env_to_demo(env, arm_key, h5_path, demo_id):
    zero_action = np.zeros(env.action_dim)

    with h5py.File(h5_path, "r") as f_h5:
        if f"data/{demo_id}/states" not in f_h5:
            return False, "No state"

        env.reset()
        env.sim.set_state_from_flattened(f_h5[f"data/{demo_id}/states"][0])
        env.sim.forward()

    for _ in range(WARMUP_STEPS):
        env.step(zero_action)

    robot = env.robots[0]
    if hasattr(robot, "composite_controller") and arm_key in robot.composite_controller.part_controllers:
        arm_ctrl = robot.composite_controller.part_controllers[arm_key]
        try:
            arm_ctrl.goal_pos = np.array(env._get_observations()["robot0_eef_pos"])
        except Exception:
            pass

    return True, ""


def get_eef_pos(env):
    obs = env._get_observations()
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()


def get_gt_endpoint_positions_from_meta(task, demo_id, cfg, meta_dir):
    """
    For each comparison step, reset env to the B-frame state and read robot0_eef_pos.
    """
    frame_indices, reason = load_meta_frame_indices(meta_dir, demo_id, cfg["compare_steps"])
    if frame_indices is None:
        return None, reason

    positions = []
    env, arm_key = make_env(cfg["env_name"])

    try:
        with h5py.File(cfg["h5_path"], "r") as f_h5:
            if f"data/{demo_id}/states" not in f_h5:
                env.close()
                return None, "No states in h5"

            states = f_h5[f"data/{demo_id}/states"]
            for step_name in cfg["compare_steps"]:
                idx = frame_indices[step_name]
                if idx < 0 or idx >= len(states):
                    env.close()
                    return None, f"Frame index out of range: {demo_id} {step_name} {idx}"

                env.reset()
                env.sim.set_state_from_flattened(states[idx])
                env.sim.forward()
                positions.append(get_eef_pos(env))

        env.close()
        return np.asarray(positions, dtype=np.float64), ""

    except Exception as e:
        try:
            env.close()
        except Exception:
            pass
        return None, str(e)


def unnormalize_action(action_norm, stats):
    min_val, max_val = stats["min"], stats["max"]
    action = (action_norm + 1) / 2
    return action * (max_val - min_val) + min_val


def predict_action_chunk(model, obs_img, goal_img, scheduler, device):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    obs_tensor = tf(obs_img).unsqueeze(0).to(device)
    goal_tensor = tf(goal_img).unsqueeze(0).to(device)

    noisy_action = torch.randn((1, ACTION_CHUNK_SIZE, 7), device=device)

    with torch.no_grad():
        cond = model.get_latents(obs_tensor, goal_tensor)
        for t in scheduler.timesteps:
            timestep_in = torch.tensor([t], device=device)
            noise_pred = model(noisy_action, timestep_in, global_cond=cond)
            noisy_action = scheduler.step(noise_pred, t, noisy_action).prev_sample

    return noisy_action[0]


def load_model(ckpt_path, device):
    vision_encoder = VisionEncoder(model_name="resnet18", pretrained=False)
    model = DiffusionPolicyGCBC(vision_encoder)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model


def rollout_endpoint_positions(task, demo_id, cfg, subgoal_type, model, stats, scheduler, device,
                               max_steps_per_subgoal, exec_horizon):
    """
    Run rollout and record eef position at the end of each non-final phase.
    For lift: after step1.
    For can/square: after step1 and step2.
    """
    env, arm_key = make_env(cfg["env_name"])

    try:
        ok, reason = reset_env_to_demo(env, arm_key, cfg["h5_path"], demo_id)
        if not ok:
            env.close()
            return None, False, 0, reason

        subgoals = []
        for step_name in cfg["subgoal_sequence"]:
            p = get_subgoal_path(task, subgoal_type, demo_id, step_name, cfg)
            if not os.path.exists(p):
                env.close()
                return None, False, 0, f"Missing subgoal: {p}"
            subgoals.append(Image.open(p).convert("RGB"))

        pred_positions = []
        success = False
        total_steps = 0

        for phase_idx, step_name in enumerate(cfg["subgoal_sequence"]):
            subgoal_img = subgoals[phase_idx]
            is_final_phase = (phase_idx == len(cfg["subgoal_sequence"]) - 1)
            phase_steps = 0

            while phase_steps < max_steps_per_subgoal:
                obs = env._get_observations()
                model_input_img = Image.fromarray(np.flip(obs["agentview_image"], 0)).convert("RGB")

                raw_actions = predict_action_chunk(model, model_input_img, subgoal_img, scheduler, device)
                actions = unnormalize_action(raw_actions.cpu().numpy(), stats)

                steps_to_exec = min(exec_horizon, len(actions))
                if phase_steps + steps_to_exec > max_steps_per_subgoal:
                    steps_to_exec = max_steps_per_subgoal - phase_steps

                for i in range(steps_to_exec):
                    env.step(actions[i])
                    total_steps += 1
                    phase_steps += 1

                    if is_final_phase and env._check_success():
                        success = True
                        break

                if success:
                    break

            # record endpoint after non-final stages only
            if step_name in cfg["compare_steps"]:
                pred_positions.append(get_eef_pos(env))

            if success:
                break

        env.close()

        if len(pred_positions) != len(cfg["compare_steps"]):
            return None, success, total_steps, f"Pred endpoint count mismatch: {len(pred_positions)} vs {len(cfg['compare_steps'])}"

        return np.asarray(pred_positions, dtype=np.float64), success, total_steps, ""

    except Exception as e:
        try:
            env.close()
        except Exception:
            pass
        return None, False, 0, str(e)


def endpoint_mse(pred_positions, gt_positions):
    pred_positions = np.asarray(pred_positions, dtype=np.float64)
    gt_positions = np.asarray(gt_positions, dtype=np.float64)

    if pred_positions.shape != gt_positions.shape:
        raise ValueError(f"Shape mismatch: pred {pred_positions.shape}, gt {gt_positions.shape}")

    diff = pred_positions - gt_positions
    mse_xyz = float(np.mean(diff ** 2))
    mean_l2 = float(np.mean(np.linalg.norm(diff, axis=1)))
    per_stage_l2 = np.linalg.norm(diff, axis=1).tolist()
    return mse_xyz, mean_l2, per_stage_l2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["lift", "can", "square"])
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument("--start-demo", type=int, default=0)
    parser.add_argument("--meta-dir", default=None, help="Override meta directory.")
    parser.add_argument("--gt-ckpt-path", default=None)
    parser.add_argument("--inf-ckpt-path", default=None)
    parser.add_argument("--mix-ckpt-path", default=None)
    parser.add_argument("--max-steps-per-subgoal", type=int, default=None)
    parser.add_argument("--exec-horizon", type=int, default=EXEC_HORIZON)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = TASK_CONFIG[args.task]
    meta_dir = args.meta_dir or cfg["meta_dir"]
    max_steps_per_subgoal = args.max_steps_per_subgoal or cfg["max_steps_per_subgoal"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_ckpt = args.gt_ckpt_path or latest_ckpt(cfg["ckpt_dirs"]["gt"])
    inf_ckpt = args.inf_ckpt_path or latest_ckpt(cfg["ckpt_dirs"]["inf"])
    mix_ckpt = args.mix_ckpt_path or latest_ckpt(cfg["ckpt_dirs"]["mix"])

    if args.output_json is None:
        args.output_json = f"gcbc/stage_endpoint_mse_{args.task}_first{args.num_demos}.json"
    if args.output_csv is None:
        args.output_csv = f"gcbc/stage_endpoint_mse_{args.task}_first{args.num_demos}.csv"

    print("=" * 80)
    print("Stage endpoint MSE")
    print(f"Task: {args.task}")
    print(f"Device: {device}")
    print(f"H5: {cfg['h5_path']}")
    print(f"Meta dir: {meta_dir}")
    print(f"Stats: {cfg['stats_path']}")
    print(f"Subgoal sequence: {cfg['subgoal_sequence']}")
    print(f"Compare steps: {cfg['compare_steps']}")
    print(f"Max steps per subgoal: {max_steps_per_subgoal}")
    print(f"Exec horizon: {args.exec_horizon}")
    print(f"GT ckpt: {gt_ckpt}")
    print(f"INF ckpt: {inf_ckpt}")
    print(f"MIX ckpt: {mix_ckpt}")
    print(f"Output JSON: {args.output_json}")
    print(f"Output CSV: {args.output_csv}")
    print("=" * 80)

    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(
            f"Meta dir not found: {meta_dir}\n"
            f"Please pass --meta-dir /path/to/meta if this task's meta is stored elsewhere."
        )

    stats = np.load(cfg["stats_path"], allow_pickle=True).item()

    scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    print("Loading models...")
    models = {
        "gt": load_model(gt_ckpt, device),
        "inf": load_model(inf_ckpt, device),
        "mix": load_model(mix_ckpt, device),
    }

    ckpt_map = {
        "gt": gt_ckpt,
        "inf": inf_ckpt,
        "mix": mix_ckpt,
    }

    demo_ids = [f"demo_{i}" for i in range(args.start_demo, args.start_demo + args.num_demos)]

    raw_results = []
    agg = {}

    for subgoal_type in ["gt", "inf"]:
        for model_type in ["gt", "inf", "mix"]:
            key = f"{model_type}_model_{subgoal_type}_subgoal"
            agg[key] = {
                "mse_values": [],
                "mean_l2_values": [],
                "success": 0,
                "evaluated": 0,
                "skipped": 0,
            }

    start_time = time.time()

    for demo_idx, demo_id in enumerate(demo_ids):
        print(f"\n========== Demo {demo_idx + 1}/{len(demo_ids)}: {demo_id} ==========")

        gt_positions, reason = get_gt_endpoint_positions_from_meta(args.task, demo_id, cfg, meta_dir)
        if gt_positions is None:
            print(f"Skip {demo_id}: cannot get GT endpoint positions: {reason}")
            for subgoal_type in ["gt", "inf"]:
                for model_type in ["gt", "inf", "mix"]:
                    agg[f"{model_type}_model_{subgoal_type}_subgoal"]["skipped"] += 1
            continue

        for subgoal_type in ["gt", "inf"]:
            if not has_all_subgoals(args.task, subgoal_type, demo_id, cfg):
                print(f"  subgoal={subgoal_type}: missing subgoals, skip all models")
                for model_type in ["gt", "inf", "mix"]:
                    agg[f"{model_type}_model_{subgoal_type}_subgoal"]["skipped"] += 1
                continue

            for model_type in ["gt", "inf", "mix"]:
                key = f"{model_type}_model_{subgoal_type}_subgoal"
                print(f"  Rollout {key} ... ", end="", flush=True)

                # Reset seed before each rollout for fair model comparison.
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED)

                pred_positions, success, steps, err = rollout_endpoint_positions(
                    task=args.task,
                    demo_id=demo_id,
                    cfg=cfg,
                    subgoal_type=subgoal_type,
                    model=models[model_type],
                    stats=stats,
                    scheduler=scheduler,
                    device=device,
                    max_steps_per_subgoal=max_steps_per_subgoal,
                    exec_horizon=args.exec_horizon,
                )

                if pred_positions is None:
                    print(f"SKIP ({err})")
                    agg[key]["skipped"] += 1
                    raw_results.append({
                        "demo_id": demo_id,
                        "subgoal_type": subgoal_type,
                        "model_type": model_type,
                        "evaluated": False,
                        "success": False,
                        "mse_xyz": None,
                        "mean_l2": None,
                        "per_stage_l2": None,
                        "gt_positions": gt_positions.tolist(),
                        "pred_positions": None,
                        "steps": steps,
                        "error": err,
                        "ckpt_path": ckpt_map[model_type],
                    })
                    continue

                mse_xyz, mean_l2, per_stage_l2 = endpoint_mse(pred_positions, gt_positions)

                agg[key]["evaluated"] += 1
                agg[key]["mse_values"].append(mse_xyz)
                agg[key]["mean_l2_values"].append(mean_l2)
                agg[key]["success"] += int(success)

                print(
                    f"MSE={mse_xyz:.8f}, "
                    f"meanL2={mean_l2:.6f}, "
                    f"perStageL2={per_stage_l2}, "
                    f"success={success}, steps={steps}"
                )

                raw_results.append({
                    "demo_id": demo_id,
                    "subgoal_type": subgoal_type,
                    "model_type": model_type,
                    "evaluated": True,
                    "success": bool(success),
                    "mse_xyz": mse_xyz,
                    "mean_l2": mean_l2,
                    "per_stage_l2": per_stage_l2,
                    "gt_positions": gt_positions.tolist(),
                    "pred_positions": pred_positions.tolist(),
                    "steps": int(steps),
                    "error": "",
                    "ckpt_path": ckpt_map[model_type],
                })

    summary = {}
    for key, v in agg.items():
        mse_arr = np.array(v["mse_values"], dtype=np.float64)
        l2_arr = np.array(v["mean_l2_values"], dtype=np.float64)
        evaluated = v["evaluated"]

        summary[key] = {
            "evaluated": int(evaluated),
            "skipped": int(v["skipped"]),
            "success": int(v["success"]),
            "success_rate": float(v["success"] / evaluated) if evaluated > 0 else 0.0,
            "mse_mean": float(np.mean(mse_arr)) if evaluated > 0 else None,
            "mse_std": float(np.std(mse_arr)) if evaluated > 0 else None,
            "mean_l2_mean": float(np.mean(l2_arr)) if evaluated > 0 else None,
            "mean_l2_std": float(np.std(l2_arr)) if evaluated > 0 else None,
        }

    report = {
        "metric": "stage_endpoint_eef_position_mse",
        "task": args.task,
        "num_demos": args.num_demos,
        "start_demo": args.start_demo,
        "demo_ids": demo_ids,
        "h5_path": cfg["h5_path"],
        "meta_dir": meta_dir,
        "stats_path": cfg["stats_path"],
        "subgoal_sequence": cfg["subgoal_sequence"],
        "compare_steps": cfg["compare_steps"],
        "max_steps_per_subgoal": max_steps_per_subgoal,
        "exec_horizon": args.exec_horizon,
        "ckpts": ckpt_map,
        "summary": summary,
        "raw_results": raw_results,
        "elapsed_sec": time.time() - start_time,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "key", "evaluated", "skipped", "success", "success_rate",
            "mse_mean", "mse_std", "mean_l2_mean", "mean_l2_std"
        ])
        for key, v in summary.items():
            writer.writerow([
                key,
                v["evaluated"],
                v["skipped"],
                v["success"],
                v["success_rate"],
                v["mse_mean"],
                v["mse_std"],
                v["mean_l2_mean"],
                v["mean_l2_std"],
            ])

    print("\n" + "=" * 80)
    print("Stage Endpoint MSE Summary")
    for key, v in summary.items():
        print(
            f"{key}: "
            f"MSE={v['mse_mean']}, "
            f"meanL2={v['mean_l2_mean']}, "
            f"success={v['success']}/{v['evaluated']} ({v['success_rate'] * 100:.1f}%)"
        )
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
