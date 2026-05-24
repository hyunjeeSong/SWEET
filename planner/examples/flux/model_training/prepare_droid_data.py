import json
import csv
import os

INPUT_JSON = "./data/prompt_flux/RPL_train700_image_prompt_list.json"
OUTPUT_CSV = "./data/prompt_flux/training_meta_droid.csv"

RULE_PROMPT = "Follow the semi-transparent motion marker in the image: an arrow indicates movement direction, while a circle indicates an in-place action. Color indicates gripper-state transition: green = open to closed, blue = closed to closed, yellow = closed to open, red = open to open."

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Not found {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for item in data:
        full_prompt = f"{RULE_PROMPT} {item['prompt']}"
        records.append({
            "image": item['final_path'],
            "kontext_images": item['first_arrow_path'],
            "prompt": full_prompt
        })

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "kontext_images", "prompt"])
        writer.writeheader()
        writer.writerows(records)
        
    print(f"✅ DROID Image Training CSV generated: {len(records)} samples -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
