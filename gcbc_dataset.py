import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DROID_GCBC_Dataset(Dataset):
    def __init__(self, 
                 start_img_dir, 
                 goal_img_dir, 
                 action_h5_dir, 
                 stats_path=None,
                 transform=None,
                 action_seq_len=16):
        
        self.start_img_dir = start_img_dir
        self.goal_img_dir = goal_img_dir
        self.action_h5_dir = action_h5_dir
        self.transform = transform
        self.action_seq_len = action_seq_len

        self.action_files = sorted(glob.glob(os.path.join(action_h5_dir, "*.h5")))
        
        self.valid_indices = []
        print(f"[Dataset] Scanned {len(self.action_files)} H5 files. Verifying pairs...")
        
        for h5_path in self.action_files:
            basename = os.path.basename(h5_path)
            img_name = os.path.splitext(basename)[0] + ".png"
            
            start_img_path = os.path.join(start_img_dir, img_name)
            goal_img_path = os.path.join(goal_img_dir, img_name)
            
            if os.path.exists(start_img_path) and os.path.exists(goal_img_path):
                self.valid_indices.append({
                    "action_path": h5_path,
                    "start_img_path": start_img_path,
                    "goal_img_path": goal_img_path
                })
        
        print(f"[Dataset] Verified {len(self.valid_indices)} valid triplets.")

        self.stats = None
        if stats_path and os.path.exists(stats_path):
            print(f"[Dataset] Loading stats from {stats_path}")
            self.stats = np.load(stats_path, allow_pickle=True).item()
            self.action_min = self.stats['min']
            self.action_max = self.stats['max']
            self.action_max = np.where(self.action_max == self.action_min, self.action_max + 1, self.action_max)

    def normalize_action(self, action):
        if self.stats is None:
            return action
        action_norm = (action - self.action_min) / (self.action_max - self.action_min)
        action_norm = action_norm * 2 - 1
        return action_norm

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        item = self.valid_indices[idx]
        
        # 1. Load Images
        try:
            start_img = Image.open(item["start_img_path"]).convert("RGB")
            goal_img = Image.open(item["goal_img_path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {item['start_img_path']} or goal")
            raise e

        if self.transform:
            start_img = self.transform(start_img)
            goal_img = self.transform(goal_img)

        # 2. Load Action (Move logic inside try block)
        try:
            with h5py.File(item["action_path"], "r", swmr=True, libver='latest') as f:
                if "action/cartesian_position" in f and "action/gripper_position" in f:
                    cart_pos = f["action/cartesian_position"][:] 
                    gripper_pos = f["action/gripper_position"][:]
                else:
                    raise ValueError(f"Invalid H5 keys in: {item['action_path']}")

                if gripper_pos.ndim == 1:
                    gripper_pos = gripper_pos[:, None]
                
                # Raw Action
                action = np.concatenate([cart_pos, gripper_pos], axis=-1)
                
            
            # 3. Normalize
            if self.stats is not None:
                action = self.normalize_action(action)
                
            # 4. Pad / Crop Length
            curr_len = action.shape[0]
            if curr_len < self.action_seq_len:
                last_action = action[-1]
                pad_len = self.action_seq_len - curr_len
                padding = np.tile(last_action, (pad_len, 1))
                action = np.concatenate([action, padding], axis=0)
            elif curr_len > self.action_seq_len:
                action = action[:self.action_seq_len]
                
            action = torch.from_numpy(action).float()

            return {
                "obs": start_img,
                "goal": goal_img,
                "action": action
            }

        except Exception as e:
            print(f"Error loading h5: {item['action_path']}")
            raise e


class Robomimic_GCBC_Dataset(Dataset):
    def __init__(self, 
                 start_img_dir, 
                 goal_img_dir, 
                 action_dir,
                 stats_path=None,
                 transform=None,
                 action_seq_len=16):
        
        self.start_img_dir = start_img_dir
        self.goal_img_dir = goal_img_dir
        self.action_dir = action_dir
        self.transform = transform
        self.action_seq_len = action_seq_len

        self.action_files = sorted(glob.glob(os.path.join(action_dir, "*.npy")))
        
        self.valid_indices = []
        print(f"[Dataset] Scanned {len(self.action_files)} action files. Verifying pairs...")
        
        for act_path in self.action_files:
            basename = os.path.basename(act_path)     # demo_0.npy
            file_id = os.path.splitext(basename)[0]   # demo_0
            img_name = file_id + ".png"               # demo_0.png
            
            start_img_path = os.path.join(start_img_dir, img_name)
            goal_img_path = os.path.join(goal_img_dir, img_name)
            
            if os.path.exists(start_img_path) and os.path.exists(goal_img_path):
                self.valid_indices.append({
                    "action_path": act_path,
                    "start_img_path": start_img_path,
                    "goal_img_path": goal_img_path
                })
        
        print(f"[Dataset] Verified {len(self.valid_indices)} valid triplets.")

        self.stats = None
        if stats_path and os.path.exists(stats_path):
            print(f"[Dataset] Loading stats from {stats_path}")
            self.stats = np.load(stats_path, allow_pickle=True).item()
            self.action_min = self.stats['min']  # shape (7,)
            self.action_max = self.stats['max']
            diff = self.action_max - self.action_min
            self.action_max = np.where(diff < 1e-6, self.action_min + 1.0, self.action_max)
        else:
            print("[Dataset] Warning: No stats_path provided. Actions will NOT be normalized.")

    def normalize_action(self, action):
        if self.stats is None:
            return action
        action_norm = (action - self.action_min) / (self.action_max - self.action_min)
        action_norm = action_norm * 2 - 1
        return action_norm

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        item = self.valid_indices[idx]
        
        # 1. Load Images
        try:
            start_img = Image.open(item["start_img_path"]).convert("RGB")
            goal_img = Image.open(item["goal_img_path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {item['start_img_path']}")
            raise e

        if self.transform:
            start_img = self.transform(start_img)
            goal_img = self.transform(goal_img)

        # 2. Load Action form .npy
        try:
            action = np.load(item["action_path"]) # Shape (T, 7)
        except Exception as e:
            print(f"Error loading npy: {item['action_path']}")
            raise e

        # 3. Normalize
        if self.stats is not None:
            action = self.normalize_action(action)

        # 4. Pad / Crop Length
        curr_len = action.shape[0]
        if curr_len < self.action_seq_len:
            last_action = action[-1]
            pad_len = self.action_seq_len - curr_len
            padding = np.tile(last_action, (pad_len, 1))
            action = np.concatenate([action, padding], axis=0)
        elif curr_len > self.action_seq_len:
            action = action[:self.action_seq_len]
            
        action = torch.from_numpy(action).float()

        return {
            "obs": start_img,
            "goal": goal_img,
            "action": action
        }
