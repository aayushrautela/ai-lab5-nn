# train.py

import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# --- 0. Prep output folders ---
os.makedirs("plots", exist_ok=True)
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

def get_loaders(batch_size):
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set,   batch_size=batch_size),
        DataLoader(test_set,  batch_size=batch_size),
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

# --- 3. Training/Eval function ---
def train_and_eval(hidden_layers, lr, bs, loss_fn, epochs=10, device="cpu"):
    train_loader, val_loader, _ = get_loaders(bs)
    model = MLP(28*28, hidden_layers, 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    for e in range(epochs):
        # -- training pass --
        model.train()
        total_loss, total_correct, total_seen = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_seen += X.size(0)

        history["train_loss"].append(total_loss / total_seen)
        history["train_acc"].append(total_correct / total_seen)

        # -- validation pass --
        model.eval()
        val_loss, val_correct, val_seen = 0, 0, 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                lv = model(Xv)
                lval = loss_fn(lv, yv)

                val_loss += lval.item() * Xv.size(0)
                val_correct += (lv.argmax(dim=1) == yv).sum().item()
                val_seen += Xv.size(0)

        history["val_loss"].append(val_loss / val_seen)
        history["val_acc"].append(val_correct / val_seen)

        # --- progress print ---
        print(f"Epoch {e+1}/{epochs}  "
              f"Train loss {history['train_loss'][-1]:.4f}, acc {history['train_acc'][-1]:.2%}  "
              f"Val   loss {history['val_loss'][-1]:.4f}, acc {history['val_acc'][-1]:.2%}",
              flush=True)

    return history

# --- 4. Hyper-parameter grid & losses ---
loss_fns = {
    "CE":  nn.CrossEntropyLoss(),
    "MSE": lambda logits,y: nn.MSELoss()(
                torch.softmax(logits, dim=1),
                nn.functional.one_hot(y, 10).float()
            ),
    "MAE": lambda logits,y: nn.L1Loss()(
                torch.softmax(logits, dim=1),
                nn.functional.one_hot(y, 10).float()
            )
}

learning_rates = [1e-1, 1e-2, 1e-3]
batch_sizes    = [1, 32, 256]
hidden_configs = [[], [128], [256, 128]]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", device)

results = []

for lr, bs, hidden, lf_name in itertools.product(
        learning_rates, batch_sizes, hidden_configs, loss_fns.keys()
    ):

    print(f"\n--- Testing LR={lr}, BS={bs}, HL={hidden}, Loss={lf_name} ---")
    history = train_and_eval(hidden, lr, bs, loss_fns[lf_name],
                              epochs=10, device=device)

    # save a sample plot for this config
    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"],   label="val   loss")
    plt.title(f"LR={lr}, BS={bs}, HL={hidden}, Loss={lf_name}")
    plt.legend()
    plt.savefig(f"plots/lr{lr}_bs{bs}_hl{len(hidden)}_{lf_name}.png")
    plt.close()

    results.append({
        "lr": lr, "bs": bs, "hidden": hidden, "loss_fn": lf_name,
        "train_acc": history["train_acc"][-1],
        "val_acc":   history["val_acc"][-1]
    })

# --- 5. Report best ---
best = max(results, key=lambda r: r["val_acc"])
print("\n=== BEST CONFIG ===")
print(best)
