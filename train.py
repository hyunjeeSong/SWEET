# python train.py \
#   --task lift \
#   --goal-type mix \
#   --batch-size 32 \
#   --num-workers 4

# python gcbc/train.py \
#   --task lift \
#   --goal-type mix \
#   --data-root /path/to/robomimic_lift_manual_dataset \
#   --stats-file /path/to/robomimic_lift_manual_stats.npy \
#   --save-root /path/to/checkpoints

import os
import argparse
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys

# Project path
GCBC_ROOT = "gcbc"
if GCBC_ROOT not in sys.path:
    sys.path.append(GCBC_ROOT)

from gcbc_dataset import Robomimic_GCBC_Dataset
from gcbc_model import VisionEncoder, DiffusionPolicyGCBC


TASK_CONFIG = {
    "lift": {
        "data_root": "data/robomimic_lift_manual_dataset",
        "stats_file": "gcbc/robomimic_lift_manual_stats.npy",
        "save_root": "gcbc",
        "default_epochs": {
            "gt": 1000,
            "inf": 1000,
            "mix": 500,
        },
        "save_every": {
            "gt": 100,
            "inf": 100,
            "mix": 50,
        },
    },
    "can": {
        "data_root": "data/robomimic_can_manual_dataset",
        "stats_file": "gcbc/robomimic_can_manual_stats.npy",
        "save_root": "gcbc",
        # Keep current script behavior unchanged
        "default_epochs": {
            "gt": 1000,
            "inf": 1000,
            "mix": 500,
        },
        "save_every": {
            "gt": 100,
            "inf": 100,
            "mix": 50,
        },
    },
    "square": {
        "data_root": "data/robomimic_square_manual_dataset",
        "stats_file": "gcbc/robomimic_square_manual_stats.npy",
        "save_root": "gcbc",
        "default_epochs": {
            "gt": 1000,
            "inf": 1000,
            "mix": 500,
        },
        "save_every": {
            "gt": 100,
            "inf": 100,
            "mix": 50,
        },
    },
}


def get_dirs(data_root, goal_type):
    if goal_type == "gt":
        return {
            "start_dir": os.path.join(data_root, "images_first_H16"),
            "goal_dir": os.path.join(data_root, "images_goal_gt_H16"),
            "action_dir": os.path.join(data_root, "action_H16"),
        }
    elif goal_type == "inf":
        return {
            "start_dir": os.path.join(data_root, "images_first_H16"),
            "goal_dir": os.path.join(data_root, "images_goal_inf_H16"),
            "action_dir": os.path.join(data_root, "action_H16"),
        }
    elif goal_type == "mix":
        return {
            "start_dir": os.path.join(data_root, "images_first_mix_H16"),
            "goal_dir": os.path.join(data_root, "images_goal_mix_H16"),
            "action_dir": os.path.join(data_root, "action_mix_H16"),
        }
    else:
        raise ValueError(f"Unknown goal_type: {goal_type}")


def get_save_dir(save_root, task, goal_type):
    if goal_type == "gt":
        return os.path.join(save_root, f"checkpoints_robomimic_{task}_manual")
    elif goal_type == "inf":
        return os.path.join(save_root, f"checkpoints_robomimic_{task}_manual_inf")
    elif goal_type == "mix":
        return os.path.join(save_root, f"checkpoints_robomimic_{task}_manual_mix")
    else:
        raise ValueError(f"Unknown goal_type: {goal_type}")


def train(args):
    if args.task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task: {args.task}")

    cfg = TASK_CONFIG[args.task]

    data_root = args.data_root or cfg["data_root"]
    stats_file = args.stats_file or cfg["stats_file"]
    save_root = args.save_root or cfg["save_root"]
    save_dir = get_save_dir(save_root, args.task, args.goal_type)

    # data_root = cfg["data_root"]
    # stats_file = cfg["stats_file"]
    # save_dir = get_save_dir(cfg["save_root"], args.task, args.goal_type)
    os.makedirs(save_dir, exist_ok=True)

    dirs = get_dirs(data_root, args.goal_type)
    start_dir = dirs["start_dir"]
    goal_dir = dirs["goal_dir"]
    action_dir = dirs["action_dir"]

    batch_size = args.batch_size
    lr = args.lr

    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = cfg["default_epochs"][args.goal_type]

    save_every = cfg["save_every"][args.goal_type]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Goal type: {args.goal_type}")
    print(f"Data root: {data_root}")
    print(f"Start dir: {start_dir}")
    print(f"Goal dir: {goal_dir}")
    print(f"Action dir: {action_dir}")
    print(f"Stats file: {stats_file}")
    print(f"Save dir: {save_dir}")
    print(f"Batch size: {batch_size}")
    print(f"LR: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Save every: {save_every}")
    print(f"Device: {device}")
    print("=" * 80)

    for p in [start_dir, goal_dir, action_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing directory: {p}")
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Missing stats file: {stats_file}")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    print("Initializing Dataset...")
    dataset = Robomimic_GCBC_Dataset(
        start_img_dir=start_dir,
        goal_img_dir=goal_dir,
        action_dir=action_dir,
        stats_path=stats_file,
        transform=tf,
        action_seq_len=16
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print("Initializing Model...")
    vision_encoder = VisionEncoder(model_name="resnet18", pretrained=True)
    model = DiffusionPolicyGCBC(vision_encoder)
    model.to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Starting Training on {device}...")
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(
            dataloader,
            desc=f"{args.task}-{args.goal_type} Epoch {epoch + 1}/{epochs}"
        )
        total_loss = 0.0

        for batch in pbar:
            obs = batch["obs"].to(device)
            goal = batch["goal"].to(device)
            gt_action = batch["action"].to(device)

            cond_embedding = model.get_latents(obs, goal)

            noise = torch.randn_like(gt_action)
            B = gt_action.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device
            ).long()

            noisy_action = noise_scheduler.add_noise(gt_action, noise, timesteps)
            noise_pred = model(noisy_action, timesteps, global_cond=cond_embedding)

            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.6f}")

        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"gcbc_ckpt_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["lift", "can", "square"])
    parser.add_argument("--goal-type", required=True, choices=["gt", "inf", "mix"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--data-root", default=None)
    parser.add_argument("--stats-file", default=None)
    parser.add_argument("--save-root", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
