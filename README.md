# SWEET
Official Code of  SWEET: Sparse World Modeling with Image Editing for Embodied Task Execution
**SWEET** is a keyframe-based visual planning framework built on the FLUX-Kontext image editing model. Instead of predicting dense future videos, SWEET progressively generates a sequence of task-relevant keyframes and converts them into executable robot actions through a goal-conditioned action predictor.


## 🧠 Method Overview

The system consists of two complementary modules:

| Name | Description |
|---|---|
| **Image Editing Planner** | Built on FLUX-Kontext with LoRA fine-tuning. Given the current observation, a text prompt, and arrow-based spatial guidance, it predicts the next visual subgoal. |
| **Action Predictor** | A goal-conditioned Diffusion Policy that takes the current observation and the generated target keyframe as visual conditions, and predicts executable action chunks. |


## 🛠️ Installation

```bash
# 1. Clone this repository
git clone https://github.com/showlab/SWEET.git
cd SWEET

# 2. Create the Conda environment
conda env create -f environment.yml
conda activate sweet

# 3. Install DiffSynth-Studio
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
cd ..

# 4. Install diffusion_policy
git clone https://github.com/real-stanford/diffusion_policy.git

# 5. Install robomimic
git clone https://github.com/ARISE-Initiative/robomimic.git

# 6. Install robosuite
git clone https://github.com/ARISE-Initiative/robosuite.git


## 📁 Project Structure

```text
SWEET/
├── environment.yml                  # Conda environment file
├── README.md                        # Project documentation
├── gcbc/                            # Goal-conditioned low-level action predictor
│   ├── gcbc_dataset.py              # Dataset loader for GCBC / Diffusion Policy training
│   ├── gcbc_model.py                # Vision encoder + goal-conditioned Diffusion Policy
│   ├── process_manual_task_data.py  # Convert annotated Robomimic data into action chunks
│   ├── normalize_stats.py           # Compute action normalization statistics
│   ├── train.py                     # Train GT / INF / MIX action predictors
│   ├── single_inference_vis.py      # Single-demo rollout with video visualization
│   └── batch_inference_benchmark.py # Batch rollout evaluation for MSE and success rate
├── planner/                         # High-level visual planner (place it in the corresponding locations in DiffSynth-Studio)
├── data/                            # Dataset loading code
├── library/                         # Trained planner and action predictor checkpoints
├── DiffSynth-Studio/                # External image editing framework, cloned separately
├── diffusion_policy/                # External Diffusion Policy codebase, cloned separately
├── robomimic/                       # External simulation environment, cloned separately
└── robosuite/                       # External simulation environment, cloned separately






