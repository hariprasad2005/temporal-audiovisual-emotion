import torch

# -----------------------------
# FILE PATHS
# -----------------------------
files = [
    r"G:\final_phase_2\outputs\CREMA-D_to_AFEW.pt",
    r"G:\final_phase_2\outputs\CREMA-D_to_RAVDESS.pt",
    r"G:\final_phase_2\outputs\RAVDESS_to_CREMA-D.pt"
]

# -----------------------------
# LOAD AND PRINT RESULTS
# -----------------------------
for file in files:
    print("\n==============================")
    print(f"Loading: {file}")

    data = torch.load(file)

    # Extract values
    info = data["cross_dataset"]

    train = info["train"]
    test = info["test"]
    acc = info["accuracy"]
    f1 = info["f1_score"]

    print(f"Train Dataset : {train}")
    print(f"Test Dataset  : {test}")
    print(f"Accuracy      : {acc * 100:.2f}%")
    print(f"F1 Score      : {f1:.4f}")