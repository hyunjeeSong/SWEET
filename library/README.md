# 📁 LoRA Library

Due to the large size of the checkpoint files, we have compressed them into separate archive files and hosted them on [Hugging Face](https://huggingface.co/datasets/VEHwang/SWEET_data).

Before running the training code, please download the required checkpoint files, unzip them, and organize them under the current directory `library/`.

---

## 🔗 Dataset Link

**Hugging Face Homepage:** [VEHwang/SWEET_data](https://huggingface.co/datasets/VEHwang/SWEET_data)

---

## 📥 Download and Extraction Guide

Please download the required `.zip` files from the Hugging Face dataset page and extract them into the `library/` directory according to your experimental needs.

The recommended directory structure is as follows:

```text
library/
├── gcbc_library.zip
└── planner_library.zip
```

After extraction, please organize the corresponding folders according to the paths required by the training scripts.

---

## 📦 File Descriptions

| File Name | Description |
|---|---|
| `gcbc_library.zip` | LoRA weights available for the action predictor. |
| `planner_library.zip` | LoRA weights available for the image editing planner. |

---

