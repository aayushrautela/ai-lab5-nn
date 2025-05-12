# Simplified train.py for Hyperparameter Evaluation (v2)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# --- 0. Setup ---
plots_dir = "plots_one_var_at_a_time"
data_dir = "data"
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Set device (use GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- 1. Data Preparation ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.view(-1))  # Flatten 28x28 image to 784 vector
])

# Load FashionMNIST dataset (only need training portion for train/val split)
full_train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)

# Split training data into training and validation sets
train_dataset, val_dataset = random_split(full_train_dataset, [50_000, 10_000])

# --- 2. Model Definition ---
class MLP(nn.Module):
    """A simple Multi-Layer Perceptron model."""
    def __init__(self, input_dim, hidden_layers, output_dim):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- 3. Training and Evaluation Loop ---
def run_training_session(model, train_loader, val_loader, loss_fn, optimizer, epochs, current_device):
    """Trains and validates the model for a given number of epochs."""
    history = {
        "step_train_losses": [], "epoch_train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    model.to(current_device)

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss, correct_train, total_train_samples = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(current_device), labels.to(current_device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history["step_train_losses"].append(loss.item())
            total_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

        avg_epoch_train_loss = total_train_loss / total_train_samples if total_train_samples else 0
        train_accuracy = correct_train / total_train_samples if total_train_samples else 0
        history["epoch_train_loss"].append(avg_epoch_train_loss)
        history["train_acc"].append(train_accuracy)

        # --- Validation Phase ---
        model.eval()
        total_val_loss, correct_val, total_val_samples = 0.0, 0, 0
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                inputs_val, labels_val = inputs_val.to(current_device), labels_val.to(current_device)
                outputs_val = model(inputs_val)
                loss_val = loss_fn(outputs_val, labels_val)
                total_val_loss += loss_val.item() * inputs_val.size(0)
                _, predicted_val = torch.max(outputs_val.data, 1)
                correct_val += (predicted_val == labels_val).sum().item()
                total_val_samples += labels_val.size(0)

        avg_val_loss = total_val_loss / total_val_samples if total_val_samples else 0
        val_accuracy = correct_val / total_val_samples if total_val_samples else 0
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_accuracy)

        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_epoch_train_loss:.4f}, Acc: {train_accuracy:.2%} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2%}")

    return history

# --- 4. Plotting Function ---
def save_plots(history, config, test_category_name, varied_value_str):
    """Saves plots for loss and accuracy."""
    plot_subdir = os.path.join(plots_dir, test_category_name)
    os.makedirs(plot_subdir, exist_ok=True)

    hl_str = '-'.join(map(str, config['hidden'])) if config['hidden'] else '0'
    plot_filename = os.path.join(plot_subdir, f"plot_{varied_value_str}.png")
    # Simplified title
    title_base = f"LR={config['lr']}, BS={config['bs']}, HL={hl_str}, Loss={config['loss_fn_name']}"

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Training Loss per Step
    axs[0].plot(history["step_train_losses"], label="Training Loss (per step)", alpha=0.9)
    axs[0].set_title(f"Train Loss | {title_base} | Varied: {varied_value_str}", fontsize=10)
    axs[0].set_xlabel("Training Steps")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Accuracy per Epoch
    epochs_range = range(1, len(history["train_acc"]) + 1)
    axs[1].plot(epochs_range, history["train_acc"], label="Training Accuracy", marker='.')
    axs[1].plot(epochs_range, history["val_acc"], label="Validation Accuracy", marker='.')
    axs[1].set_title(f"Accuracy | {title_base} | Varied: {varied_value_str}", fontsize=10)
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)
    if history["val_acc"]:
         min_acc = min(min(history["train_acc"]), min(history["val_acc"]))
         axs[1].set_ylim(bottom=max(0, min_acc - 0.1), top=1.05)

    plt.tight_layout(pad=2.0)
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved plot: {plot_filename}")

# --- 5. Experiment Configuration ---

loss_functions = {
    "CE": nn.CrossEntropyLoss(),
    "MSE": lambda logits, labels: nn.MSELoss()(torch.softmax(logits, dim=1), nn.functional.one_hot(labels, num_classes=10).float()),
    "MAE": lambda logits, labels: nn.L1Loss()(torch.softmax(logits, dim=1), nn.functional.one_hot(labels, num_classes=10).float())
}

baseline_config = { "lr": 0.01, "bs": 32, "hidden": [128], "loss_fn_name": "CE" }

learning_rates_to_test = [0.1, 0.01, 0.001]
batch_sizes_to_test = [1, 32, 256]
hidden_configs_to_test = [[], [128], [256, 128]]
loss_fn_names_to_test = ["CE", "MSE", "MAE"]

EPOCHS_PER_RUN = 10

# --- 6. Run Experiments ---

all_results = {} # Stores history { "Category": [ {param: val, history: hist}, ... ], ... }

# Function to run a single experiment configuration
def run_experiment(config, category_name, varied_param_str):
    print(f"\n--- Running {category_name}: {varied_param_str} ---")
    print(f"Config: {config}")

    # Create DataLoaders directly here
    train_loader = DataLoader(train_dataset, batch_size=config['bs'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['bs'])

    model = MLP(input_dim=28*28, hidden_layers=config['hidden'], output_dim=10)
    loss_fn = loss_functions[config['loss_fn_name']]
    optimizer = optim.SGD(model.parameters(), lr=config['lr'])

    history = run_training_session(model, train_loader, val_loader, loss_fn, optimizer, EPOCHS_PER_RUN, device)
    save_plots(history, config, category_name, varied_param_str)
    return history

# --- Run Baseline ---
baseline_history = run_experiment(baseline_config, "Baseline", "Baseline")
all_results["Baseline"] = {"config": baseline_config, "history": baseline_history}

# --- Test Loop Function ---
def run_test_series(param_name, values_to_test, baseline_value, category_name_prefix):
    results_list = []
    category_name = f"{category_name_prefix}_Test" # Define category name once

    for test_value in values_to_test:
        if test_value == baseline_value:
            print(f"\nSkipping {param_name}={test_value} (same as baseline)")
            # Store baseline history directly for summary later
            results_list.append({param_name: test_value, "history": baseline_history, "is_baseline": True})
            continue

        current_config = baseline_config.copy()
        current_config[param_name] = test_value

        # Determine varied_str for plot filename/title
        if param_name == "loss_fn_name": varied_str = f"Loss_{test_value}"
        elif param_name == "hidden": varied_str = f"HL_{'-'.join(map(str, test_value)) if test_value else '0'}"
        else: varied_str = f"{param_name.upper()}_{test_value}"

        history = run_experiment(current_config, category_name, varied_str)
        results_list.append({param_name: test_value, "history": history, "is_baseline": False})
    return results_list

# --- Run Parameter Tests ---
all_results["LR_Test"] = run_test_series("lr", learning_rates_to_test, baseline_config["lr"], "LR")
all_results["BS_Test"] = run_test_series("bs", batch_sizes_to_test, baseline_config["bs"], "BS")
all_results["HL_Test"] = run_test_series("hidden", hidden_configs_to_test, baseline_config["hidden"], "HL")
all_results["LossFn_Test"] = run_test_series("loss_fn_name", loss_fn_names_to_test, baseline_config["loss_fn_name"], "LossFn")


# --- 7. Final Summary ---
print("\n\n--- Experiment Summary (Final Validation Accuracies) ---")

# Simpler summary print
baseline_final_acc = all_results["Baseline"]["history"]["val_acc"][-1] if all_results["Baseline"]["history"]["val_acc"] else -1.0
print(f"Baseline Val Acc: {baseline_final_acc:.2%}")

for category, results in all_results.items():
    if category == "Baseline": continue # Skip baseline dict itself

    param_key = list(results[0].keys())[0] # Infer parameter name (e.g., 'lr', 'bs')
    print(f"\n{category} Results:")
    for r in results:
        final_val_acc = r['history']['val_acc'][-1] if r['history']['val_acc'] else -1.0
        prefix = "(Baseline)" if r.get("is_baseline", False) else ""
        print(f"  {param_key}={r[param_key]} {prefix}, Final Val Acc: {final_val_acc:.2%}")


print("\nExperiment finished. Check subdirectories inside 'plots_one_var_at_a_time' for detailed graphs.")
