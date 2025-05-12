# train.py (Modified for One-Variable-at-a-Time Evaluation)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
# torch.amp for Automatic Mixed Precision
from torch.amp import autocast, GradScaler 

# --- 0. Prep output folders ---
# Base plots directory
plots_dir = "plots_one_var_at_a_time" 
os.makedirs(plots_dir, exist_ok=True)
# Data directory
os.makedirs("data", exist_ok=True)

# --- 1. Data loaders ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.view(-1))  # flatten 28×28 → 784
])

# download if needed
full_train = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
train_set, val_set = random_split(full_train, [50_000, 10_000])
test_set = datasets.FashionMNIST("data", train=False, download=True, transform=transform)

# Define device globally so get_loaders can use it
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True # Enable cuDNN benchmark mode

def get_loaders(batch_size):
    num_w = 0
    pin_mem = False
    if device == "cuda": 
        num_w = min(os.cpu_count() or 1, 4) 
        pin_mem = True

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_w, pin_memory=pin_mem),
        DataLoader(val_set,   batch_size=batch_size, num_workers=num_w, pin_memory=pin_mem),
        DataLoader(test_set,  batch_size=batch_size, num_workers=num_w, pin_memory=pin_mem),
    )

# --- 2. Flexible MLP definition ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- 3. Training/Eval function (Mostly unchanged from previous version) ---
def train_and_eval(hidden_layers, lr, bs, loss_fn, epochs=10, current_device="cpu"):
    train_loader, val_loader, _ = get_loaders(bs) 

    model = MLP(28*28, hidden_layers, 10).to(current_device)

    if int(torch.__version__.split('.')[0]) >= 2 and current_device == "cuda":
        # print("Attempting to compile model with torch.compile()...") # Optional: uncomment to see compile messages
        try:
            model = torch.compile(model)
            # print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}") # Keep error message

    optimizer = optim.SGD(model.parameters(), lr=lr)

    scaler = None
    if current_device == "cuda":
        scaler = GradScaler() # Use torch.amp.GradScaler

    history = {
        "step_train_losses": [], "epoch_train_loss": [], 
        "train_acc": [], "val_loss": [], "val_acc": []
    }

    for e in range(epochs):
        model.train()
        current_epoch_total_loss, current_epoch_correct, current_epoch_seen = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(current_device), y.to(current_device)
            optimizer.zero_grad(set_to_none=True)

            if scaler: 
                with autocast(device_type='cuda'): 
                    logits = model(X)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: 
                logits = model(X)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
            
            history["step_train_losses"].append(loss.item())
            current_epoch_total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            current_epoch_correct += (preds == y).sum().item()
            current_epoch_seen += X.size(0)

        history["epoch_train_loss"].append(current_epoch_total_loss / current_epoch_seen if current_epoch_seen > 0 else 0)
        history["train_acc"].append(current_epoch_correct / current_epoch_seen if current_epoch_seen > 0 else 0)

        model.eval()
        current_val_epoch_total_loss, current_val_correct, current_val_seen = 0, 0, 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(current_device), yv.to(current_device)
                if current_device == "cuda" and scaler: 
                    with autocast(device_type='cuda'):
                        lv = model(Xv)
                else:
                    lv = model(Xv)
                lval = loss_fn(lv, yv)
                current_val_epoch_total_loss += lval.item() * Xv.size(0)
                current_val_correct += (lv.argmax(dim=1) == yv).sum().item()
                current_val_seen += Xv.size(0)
        history["val_loss"].append(current_val_epoch_total_loss / current_val_seen if current_val_seen > 0 else 0)
        history["val_acc"].append(current_val_correct / current_val_seen if current_val_seen > 0 else 0)

        # Only print epoch summary, removed intra-batch print
        print(f"  Epoch {e+1}/{epochs}  "
              f"Train loss {history['epoch_train_loss'][-1]:.4f}, acc {history['train_acc'][-1]:.2%}  "
              f"Val loss {history['val_loss'][-1]:.4f}, acc {history['val_acc'][-1]:.2%}",
              flush=True)
    return history

# --- 4. Plotting Helper Function ---
def save_plots(history, config, test_category_name, varied_value_str):
    """ Saves plots for loss and accuracy into category-specific folders. """
    # Create subdirectory for the test category
    plot_subdir = os.path.join(plots_dir, test_category_name)
    os.makedirs(plot_subdir, exist_ok=True)

    # Create filesystem-friendly string for hidden layers
    hl_str = str(config['hidden']).replace(' ','').replace('[','').replace(']','').replace(',','-')
    if hl_str == '': hl_str = '0' 

    # Create filename incorporating the varied value
    plot_filename = os.path.join(plot_subdir, f"plot_{varied_value_str}.png")

    fig, axs = plt.subplots(2, 1, figsize=(10, 12)) 

    # Subplot 1: Training Loss per Step
    title_loss = f"Train Loss | LR={config['lr']}, BS={config['bs']}, HL={hl_str}, Loss={config['loss_fn_name']} | Varied: {varied_value_str}"
    axs[0].plot(history["step_train_losses"], label="Training Loss (per step)", alpha=0.8)
    axs[0].set_title(title_loss, fontsize=9) # Smaller font for potentially long title
    axs[0].set_xlabel("Training Steps")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: Accuracy per Epoch
    if history["train_acc"] and history["val_acc"]:
        epochs_range = range(1, len(history["train_acc"]) + 1)
        axs[1].plot(epochs_range, history["train_acc"], label="Training Accuracy", marker='o')
        axs[1].plot(epochs_range, history["val_acc"], label="Validation Accuracy", marker='o')
        title_acc = f"Accuracy | LR={config['lr']}, BS={config['bs']}, HL={hl_str}, Loss={config['loss_fn_name']} | Varied: {varied_value_str}"
        axs[1].set_title(title_acc, fontsize=9)
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        axs[1].grid(True)
    else:
        axs[1].text(0.5, 0.5, "No accuracy data to plot.", 
                    horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].set_title("Accuracy Plot")

    plt.tight_layout(pad=3.0) 
    plt.savefig(plot_filename)
    plt.close(fig) 
    print(f"Saved plot to: {plot_filename}")

# --- 5. Define Baseline and Test Parameters ---

# Loss functions dictionary (needed for baseline and tests)
loss_fns = {
    "CE":  nn.CrossEntropyLoss(),
    "MSE": lambda logits,y: nn.MSELoss()(torch.softmax(logits, dim=1), nn.functional.one_hot(y, 10).float()),
    "MAE": lambda logits,y: nn.L1Loss()(torch.softmax(logits, dim=1), nn.functional.one_hot(y, 10).float())
}

# Baseline Configuration
baseline_config = {
    "lr": 1e-2,          # 0.01
    "bs": 32,
    "hidden": [128],     # Single hidden layer with 128 neurons
    "loss_fn_name": "CE",
}
# Add the actual loss function to the baseline dict
baseline_config["loss_fn"] = loss_fns[baseline_config["loss_fn_name"]]

# Parameter ranges to test individually
learning_rates_to_test = [1e-1, 1e-2, 1e-3]
batch_sizes_to_test = [1, 32, 256]
hidden_configs_to_test = [[], [128], [256, 128]] # 0, 1, and 2 hidden layers
loss_fns_to_test = ["CE", "MSE", "MAE"] # Test based on names

# --- 6. Run Experiments ---

print(f"Running experiments on device: {device}")
EPOCHS_PER_CONFIG = 10 # Set number of epochs for all runs

# Store results (optional, mainly for programmatic analysis later)
all_results = {} 

# --- Run Baseline ---
print(f"\n--- Running Baseline Configuration ---")
print(f"Config: {baseline_config}")
baseline_history = train_and_eval(
    hidden_layers=baseline_config["hidden"],
    lr=baseline_config["lr"],
    bs=baseline_config["bs"],
    loss_fn=baseline_config["loss_fn"],
    epochs=EPOCHS_PER_CONFIG,
    current_device=device
)
save_plots(baseline_history, baseline_config, "Baseline", "Baseline")
all_results["Baseline"] = {"config": baseline_config, "history": baseline_history}

# --- Test Learning Rates ---
print("\n\n--- Testing Learning Rates (Baseline BS, HL, Loss) ---")
all_results["LR_Test"] = []
for lr_test in learning_rates_to_test:
    # Skip if this is the same as baseline to avoid re-running
    if lr_test == baseline_config["lr"] and "Baseline" in all_results:
         print(f"\nSkipping LR={lr_test} (same as baseline)")
         # Optionally copy baseline results if needed for comparison format
         all_results["LR_Test"].append({"lr": lr_test, "history": baseline_history})
         continue

    current_config = baseline_config.copy()
    current_config["lr"] = lr_test
    print(f"\nTesting LR={lr_test}")
    history = train_and_eval(current_config["hidden"], lr_test, current_config["bs"], current_config["loss_fn"], epochs=EPOCHS_PER_CONFIG, current_device=device)
    save_plots(history, current_config, "LR_Test", f"LR_{lr_test}")
    all_results["LR_Test"].append({ "lr": lr_test, "history": history })

# --- Test Batch Sizes ---
print("\n\n--- Testing Batch Sizes (Baseline LR, HL, Loss) ---")
all_results["BS_Test"] = []
for bs_test in batch_sizes_to_test:
    if bs_test == baseline_config["bs"] and "Baseline" in all_results:
         print(f"\nSkipping BS={bs_test} (same as baseline)")
         all_results["BS_Test"].append({"bs": bs_test, "history": baseline_history})
         continue
    
    current_config = baseline_config.copy()
    current_config["bs"] = bs_test
    print(f"\nTesting BS={bs_test}")
    history = train_and_eval(current_config["hidden"], current_config["lr"], bs_test, current_config["loss_fn"], epochs=EPOCHS_PER_CONFIG, current_device=device)
    save_plots(history, current_config, "BS_Test", f"BS_{bs_test}")
    all_results["BS_Test"].append({ "bs": bs_test, "history": history })

# --- Test Hidden Layers ---
print("\n\n--- Testing Hidden Layers (Baseline LR, BS, Loss) ---")
all_results["HL_Test"] = []
for hl_test in hidden_configs_to_test:
    if hl_test == baseline_config["hidden"] and "Baseline" in all_results:
         print(f"\nSkipping HL={hl_test} (same as baseline)")
         all_results["HL_Test"].append({"hl": hl_test, "history": baseline_history})
         continue

    current_config = baseline_config.copy()
    current_config["hidden"] = hl_test
    # Create string representation for saving plots
    hl_str_test = str(hl_test).replace(' ','').replace('[','').replace(']','').replace(',','-')
    if hl_str_test == '': hl_str_test = '0'
    
    print(f"\nTesting HL={hl_test}")
    history = train_and_eval(hl_test, current_config["lr"], current_config["bs"], current_config["loss_fn"], epochs=EPOCHS_PER_CONFIG, current_device=device)
    save_plots(history, current_config, "HL_Test", f"HL_{hl_str_test}")
    all_results["HL_Test"].append({ "hl": hl_test, "history": history })

# --- Test Loss Functions ---
print("\n\n--- Testing Loss Functions (Baseline LR, BS, HL) ---")
all_results["LossFn_Test"] = []
for lf_name_test in loss_fns_to_test:
    if lf_name_test == baseline_config["loss_fn_name"] and "Baseline" in all_results:
         print(f"\nSkipping Loss={lf_name_test} (same as baseline)")
         all_results["LossFn_Test"].append({"loss_fn": lf_name_test, "history": baseline_history})
         continue
         
    current_config = baseline_config.copy()
    current_config["loss_fn_name"] = lf_name_test
    current_config["loss_fn"] = loss_fns[lf_name_test] # Get the actual function
    
    print(f"\nTesting Loss Function={lf_name_test}")
    history = train_and_eval(current_config["hidden"], current_config["lr"], current_config["bs"], current_config["loss_fn"], epochs=EPOCHS_PER_CONFIG, current_device=device)
    save_plots(history, current_config, "LossFn_Test", f"Loss_{lf_name_test}")
    all_results["LossFn_Test"].append({ "loss_fn": lf_name_test, "history": history })


# --- 7. Final Summary (Optional Simple Output) ---
print("\n\n--- Experiment Summary (Final Validation Accuracies) ---")

def print_summary(test_name, results_list, param_name):
    print(f"\n{test_name} Results:")
    # Sort results if needed, or print in order tested
    for r in results_list:
        final_val_acc = r['history']['val_acc'][-1] if r['history']['val_acc'] else -1.0
        print(f"  {param_name}={r[param_name]}, Final Val Acc: {final_val_acc:.2%}")

if "Baseline" in all_results:
    baseline_val_acc = all_results["Baseline"]["history"]["val_acc"][-1] if all_results["Baseline"]["history"]["val_acc"] else -1.0
    print(f"Baseline Val Acc: {baseline_val_acc:.2%}")

if "LR_Test" in all_results: print_summary("LR Test", all_results["LR_Test"], "lr")
if "BS_Test" in all_results: print_summary("BS Test", all_results["BS_Test"], "bs")
if "HL_Test" in all_results: print_summary("HL Test", all_results["HL_Test"], "hl")
if "LossFn_Test" in all_results: print_summary("LossFn Test", all_results["LossFn_Test"], "loss_fn")

print("\nFinished. 'plots_one_var_at_a_time' for graphs.")