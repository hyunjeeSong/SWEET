# python single_inference_vis.py \
#   --task can \
#   --ckpt-type mix \
#   --subgoal-type inf \
#   --demo-id demo_0 \
#   --ckpt-path gcbc/checkpoints_robomimic_can_manual_mix/gcbc_ckpt_100.pth

import os
import glob
import argparse
import time
import numpy as np
import torch
import h5py
import imageio
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
        "gt_subgoal_root": "data/robomimic_lift_manual_dataset/images_goal_gt_H16",
        "inf_subgoal_root": "data/robomimic_lift_manual_dataset/images_goal_inf_H16",
        "ckpt_dirs": {
            "gt": "gcbc/checkpoints_robomimic_lift_manual",
            "inf": "gcbc/checkpoints_robomimic_lift_manual_inf",
            "mix": "gcbc/checkpoints_robomimic_lift_manual_mix",
        },
        "subgoal_sequence": ["step1", "step2"],
        "max_steps_per_subgoal": 48,
        "inf_name_prefix": "",
    },
    "can": {
        "env_name": "PickPlaceCan",
        "h5_path": "data/robomimic_image/can/ph/image_v15_1280.hdf5",
        "stats_path": "gcbc/robomimic_can_manual_stats.npy",
        "gt_subgoal_root": "data/robomimic/robomimic_result/can/final",
        "inf_subgoal_root": "data/robomimic_inf/robomimic_inf_flux_6000step_30infstep",
        "ckpt_dirs": {
            "gt": "gcbc/checkpoints_robomimic_can_manual",
            "inf": "gcbc/checkpoints_robomimic_can_manual_inf",
            "mix": "gcbc/checkpoints_robomimic_can_manual_mix",
        },
        "subgoal_sequence": ["step1", "step2", "step3"],
        "max_steps_per_subgoal": 64,
        "inf_name_prefix": "can_",
    },
    "square": {
        "env_name": "NutAssemblySquare",
        "h5_path": "data/robomimic_image/square/ph/image_v15_1280.hdf5",
        "stats_path": "gcbc/robomimic_square_manual_stats.npy",
        "gt_subgoal_root": "data/robomimic/robomimic_result/square/final",
        "inf_subgoal_root": "data/robomimic_inf/robomimic_inf_flux_6000step_30infstep",
        "ckpt_dirs": {
            "gt": "gcbc/checkpoints_robomimic_square_manual",
            "inf": "gcbc/checkpoints_robomimic_square_manual_inf",
            "mix": "gcbc/checkpoints_robomimic_square_manual_mix",
        },
        "subgoal_sequence": ["step1", "step2", "step3"],
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


def unnormalize_action(action_norm, stats):
    min_val, max_val = stats["min"], stats["max"]
    action = (action_norm + 1) / 2
    return action * (max_val - min_val) + min_val


def predict_action_chunk(model, obs_img, goal_img, scheduler, device):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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
            raise ValueError(f"Demo not found or no states: {demo_id}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["lift", "can", "square"])
    parser.add_argument("--ckpt-type", required=True, choices=["gt", "inf", "mix"])
    parser.add_argument("--subgoal-type", required=True, choices=["gt", "inf"])
    parser.add_argument("--demo-id", default="demo_0")
    parser.add_argument("--ckpt-path", default=None, help="Optional explicit checkpoint path. If omitted, use latest ckpt in corresponding dir.")
    parser.add_argument("--output-dir", default="gcbc/vis_results")
    parser.add_argument("--max-steps-per-subgoal", type=int, default=None)
    parser.add_argument("--exec-horizon", type=int, default=EXEC_HORIZON)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = TASK_CONFIG[args.task]
    max_steps_per_subgoal = args.max_steps_per_subgoal or cfg["max_steps_per_subgoal"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = args.ckpt_path or latest_ckpt(cfg["ckpt_dirs"][args.ckpt_type])
    os.makedirs(args.output_dir, exist_ok=True)

    output_video = os.path.join(
        args.output_dir,
        f"vis_{args.task}_{args.ckpt_type}model_{args.subgoal_type}subgoal_{args.demo_id}.mp4"
    )

    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Env: {cfg['env_name']}")
    print(f"Checkpoint type: {args.ckpt_type}")
    print(f"Subgoal type: {args.subgoal_type}")
    print(f"Demo id: {args.demo_id}")
    print(f"CKPT: {ckpt_path}")
    print(f"Stats: {cfg['stats_path']}")
    print(f"H5: {cfg['h5_path']}")
    print(f"Subgoal sequence: {cfg['subgoal_sequence']}")
    print(f"Max steps per subgoal: {max_steps_per_subgoal}")
    print(f"Exec horizon: {args.exec_horizon}")
    print(f"Output video: {output_video}")
    print(f"Device: {device}")
    print("=" * 80)

    vision_encoder = VisionEncoder(model_name="resnet18", pretrained=False)
    model = DiffusionPolicyGCBC(vision_encoder)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    stats = np.load(cfg["stats_path"], allow_pickle=True).item()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    env, arm_key = make_env(cfg["env_name"])
    reset_env_to_demo(env, arm_key, cfg["h5_path"], args.demo_id)

    subgoals_imgs = []
    for step_name in cfg["subgoal_sequence"]:
        p = get_subgoal_path(args.task, args.subgoal_type, args.demo_id, step_name, cfg)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing subgoal image: {p}")
        subgoals_imgs.append(Image.open(p).convert("RGB"))

    writer = imageio.get_writer(output_video, fps=10)

    episode_success = False
    total_env_steps = 0
    start_time = time.time()

    for phase_idx, step_name in enumerate(cfg["subgoal_sequence"]):
        subgoal_img = subgoals_imgs[phase_idx]
        is_final_phase = (phase_idx == len(cfg["subgoal_sequence"]) - 1)
        phase_steps = 0

        while phase_steps < max_steps_per_subgoal:
            obs = env._get_observations()
            model_input_img = Image.fromarray(np.flip(obs["agentview_image"], 0)).convert("RGB")

            raw_actions = predict_action_chunk(model, model_input_img, subgoal_img, noise_scheduler, device)
            actions = unnormalize_action(raw_actions.cpu().numpy(), stats)

            steps_to_exec = min(args.exec_horizon, len(actions))
            if phase_steps + steps_to_exec > max_steps_per_subgoal:
                steps_to_exec = max_steps_per_subgoal - phase_steps

            for i in range(steps_to_exec):
                obs, reward, done, info = env.step(actions[i])
                total_env_steps += 1
                phase_steps += 1

                writer.append_data(np.ascontiguousarray(np.flip(obs["agentview_image"], 0)))

                if is_final_phase and env._check_success():
                    episode_success = True
                    break

            if episode_success:
                break

        if episode_success:
            break

    writer.close()
    env.close()

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Demo: {args.demo_id}")
    print(f"Success: {episode_success}")
    print(f"Total env steps: {total_env_steps}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Output video: {output_video}")
    print("=" * 80)


if __name__ == "__main__":
    main()
