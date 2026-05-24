import os
import csv

first_frames_dir = "./data/DROID_labeled_frames_total_first_improved"
last_frames_dir = "./data/DROID_labeled_frames_total_last_improved"
output_dir = "./data/"
output_csv = os.path.join(output_dir, "metadata_improved.csv")

# ========== Prompt settings ==========
USE_EMPTY_PROMPT = False  # replace with True to use empty prompts, or False to use the default robotic prompt
ROBOT_PROMPT = "Execute the robotic arm motion from the initial position to the target position following the arrow"

# ========== create directories ==========
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "first_frames_improved"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "last_frames_improved"), exist_ok=True)

# ========== get all images (sorted by filename) ==========
first_files = sorted([f for f in os.listdir(first_frames_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
last_files = sorted([f for f in os.listdir(last_frames_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])

print(f"Found {len(first_files)} first-frame images")
print(f"Found {len(last_files)} last-frame images")

# ========== validate file names ==========
if first_files == last_files:
    print("✓ File names match perfectly, pairing by name")
    paired_files = list(zip(first_files, last_files))
else:
    print("⚠ File names don't match perfectly, pairing by index (sorted order)")
    min_len = min(len(first_files), len(last_files))
    paired_files = list(zip(first_files[:min_len], last_files[:min_len]))
    print(f"Paired files: {min_len}")

# ========== create CSV ==========
valid_count = 0
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image', 'prompt', 'kontext_images'])
    
    for i, (first_file, last_file) in enumerate(paired_files):
        # source paths
        first_src = os.path.join(first_frames_dir, first_file)
        last_src = os.path.join(last_frames_dir, last_file)
        
        # validate source files exist
        if not os.path.exists(first_src):
            print(f"⚠ Warning: First-frame file does not exist {first_src}")
            continue
        if not os.path.exists(last_src):
            print(f"⚠ Warning: Last-frame file does not exist {last_src}")
            continue
        
        # target paths
        first_dst = os.path.join(output_dir, "first_frames_improved", first_file)
        last_dst = os.path.join(output_dir, "last_frames_improved", last_file)
        
        # create symlinks if they don't already exist
        if not os.path.exists(first_dst):
            os.symlink(first_src, first_dst)
        if not os.path.exists(last_dst):
            os.symlink(last_src, last_dst)
        
        # create prompt
        if USE_EMPTY_PROMPT:
            prompt = ""
        else:
            prompt = ROBOT_PROMPT
        
        # write CSV row
        writer.writerow([
            f"last_frames_improved/{last_file}",
            prompt,
            f"first_frames_improved/{first_file}"
        ])
        
        valid_count += 1
        if (i + 1) % 100 == 0:
            print(f"Processed {valid_count}/{len(paired_files)} pairs of images")

print(f"\nDataset preparation complete!")
print(f"Valid pairs: {valid_count} pairs of images")
print(f"CSV file: {output_csv}")
print(f"Dataset directory: {output_dir}")

# ========== Validate CSV ==========
print("\nValidating first 5 rows of CSV:")
with open(output_csv, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 6:
            print(line.strip())
