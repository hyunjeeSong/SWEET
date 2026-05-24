import json
import csv
import os

INPUT_JSON = "./data/prompt_flux/robomimic_prompt_list_train.json"
OUTPUT_CSV = "./data/prompt_flux/training_meta.csv"

# rule prompt
RULE_PROMPT = "Follow the semi-transparent motion marker in the image: an arrow indicates movement direction, while a circle indicates an in-place action. Color indicates gripper-state transition: green = open to closed, blue = closed to closed, yellow = closed to open, red = open to open."

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Not found {INPUT_JSON}")
        return

    print(f"Reading {INPUT_JSON}...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare CSV records
    # DiffSynth UnifiedDataset format: image (final state), kontext_images (first state with arrow), prompt (text instruction)
    records = []
    
    for i, item in enumerate(data):
        # Basic validation
        if 'final_path' not in item or 'first_arrow_path' not in item or 'prompt' not in item:
            print(f"Skipping item {i}, missing keys.")
            continue
            
        # Construct full prompt
        full_prompt = f"{RULE_PROMPT} {item['prompt']}"
        
        records.append({
            "image": item['final_path'],          # Target: Ground Truth (Final state)
            "kontext_images": item['first_arrow_path'], # Input: Condition (First state with arrow)
            "prompt": full_prompt
        })

    # Write to CSV
    if records:
        fieldnames = ["image", "kontext_images", "prompt"]
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        
        print(f"✅ Successfully converted {len(records)} samples.")
        print(f"   Output saved to: {OUTPUT_CSV}")
        print("   Sample Prompt:", records[0]['prompt'])
    else:
        print("No valid records found.")

if __name__ == "__main__":
    main()
