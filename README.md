# Lab 5: Artificial Neural Networks (Variant 4)

**Author:** Aayush Rautela
**Course:** Introduction to Artificial Intelligence

## Description

Implements a PyTorch Multilayer Perceptron (MLP) for FashionMNIST image classification (Lab 5, Variant 4). The script evaluates the impact of different hyperparameters (Learning Rate, Batch Size, Hidden Layers/Width, Loss Function) using a **"one variable at a time"** strategy relative to a defined baseline configuration.

## Key Features

* Trains an MLP using mini-batch SGD on FashionMNIST (auto-downloads).
* Splits data into training/validation sets.
* Evaluates the effect of varying one hyperparameter category at a time.
* Generates plots: training loss per step, train/validation accuracy per epoch.
* Saves plots into category-specific subfolders (e.g., `plots_one_var_at_a_time/LR_Test/`).
* Supports CUDA GPU acceleration (incl. optional `torch.compile` and AMP).

## Setup

1.  **Prerequisites:** Python 3 (3.10+ recommended), `pip`.
2.  **Get Code:** Download or clone `train.py` and `requirements.txt`.
3.  **Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux
    ```
4.  **Install PyTorch:** Install **first** using the official command builder for your system: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) (Select Pip, Python, CPU/CUDA).
5.  **Install Other Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running

1.  Activate the virtual environment (`source venv/bin/activate`).
2.  Run the script:
    ```bash
    python train.py
    ```
3.  The script executes the baseline run, followed by test runs for each hyperparameter category (approx. 9 runs total). Console output shows epoch progress.

## Configuration

Key parameters can be adjusted within `train.py`:
* `baseline_config`: Dictionary defining the default settings.
* `*_to_test` lists: Define the values to test for each hyperparameter category.
* `EPOCHS_PER_CONFIG`: Sets the number of training epochs for each run.

## Output

* **Console:** Epoch-level training progress and a final summary.
* **Plots:** PNG plot files saved in `plots_one_var_at_a_time/` subdirectories (Baseline, LR_Test, BS_Test, etc.). Each plot shows:
    * Top: Training loss per step.
    * Bottom: Training & Validation accuracy per epoch.