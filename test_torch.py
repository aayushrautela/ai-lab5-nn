# test_torch.py

import torch

def main():
    print(f"PyTorch version: {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA available:   {cuda}")
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Using device:     {device}")

    # Quick tensor ops
    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 4, device=device)
    z = x @ y.T
    print("Matrix‐mul OK, example output:")
    print(z)

if __name__ == "__main__":
    main()
