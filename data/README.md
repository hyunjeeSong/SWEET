# 📁 Dataset Directory

Due to the large size of the dataset files, we have compressed them into separate archive files and hosted them on [Hugging Face](https://huggingface.co/datasets/VEHwang/SWEET_data).

Before running the training code, please download the required dataset files, unzip them, and organize them under the current directory `data/`.

---

## 🔗 Dataset Link

**Hugging Face Homepage:** [VEHwang/SWEET_data](https://huggingface.co/datasets/VEHwang/SWEET_data)

---

## 📥 Download and Extraction Guide

Please download the required `.zip` files from the Hugging Face dataset page and extract them into the `data/` directory according to your experimental needs.

The recommended directory structure is as follows:

```text
data/
├── DROID_labeled_for_flux_test50_seen.zip
├── DROID_labeled_for_flux_test50_unseen.zip
├── DROID_labeled_for_flux_train700_part1.zip
├── DROID_labeled_for_flux_train700_part2.zip
├── DROID_labeled_for_wan.zip
├── gcbc_library.zip
├── planner_library.zip
├── prompt_flux.zip
├── prompt_wan.zip
├── robomimic_labeled_for_flux_can.zip
├── robomimic_labeled_for_flux_lift.zip
├── robomimic_labeled_for_flux_square.zip
└── robomimic_labeled_for_wan.zip
```

After extraction, please organize the corresponding folders according to the paths required by the training scripts.

---

## 📦 File Descriptions

| File Name | Description |
|---|---|
| `DROID_labeled_for_flux_test50_seen.zip` | DROID test set for Flux, containing samples from seen environments. |
| `DROID_labeled_for_flux_test50_unseen.zip` | DROID test set for Flux, containing samples from unseen environments. |
| `DROID_labeled_for_flux_train700_part1.zip` | The first part of the DROID training set for Flux. It should be merged with `part2`. |
| `DROID_labeled_for_flux_train700_part2.zip` | The second part of the DROID training set for Flux. It should be merged with `part1`. |
| `DROID_labeled_for_wan.zip` | DROID training and test sets used by WAN. |
| `gcbc_library.zip` | LoRA weights available for the action predictor. |
| `planner_library.zip` | LoRA weights available for the image editing planner. |
| `prompt_flux.zip` | Prompt list used for Flux training. |
| `prompt_wan.zip` | Prompt list used for WAN training. |
| `robomimic_labeled_for_flux_can.zip` | Robomimic `can` task dataset used by Flux. The three task datasets should be merged and split into test/training sets at a ratio of 1:9. |
| `robomimic_labeled_for_flux_lift.zip` | Robomimic `lift` task dataset used by Flux. The three task datasets should be merged and split into test/training sets at a ratio of 1:9. |
| `robomimic_labeled_for_flux_square.zip` | Robomimic `square` task dataset used by Flux. The three task datasets should be merged and split into test/training sets at a ratio of 1:9. |
| `robomimic_labeled_for_wan.zip` | Robomimic dataset used by WAN. The three task datasets should also be merged and split into test/training sets at a ratio of 1:9. |

---

