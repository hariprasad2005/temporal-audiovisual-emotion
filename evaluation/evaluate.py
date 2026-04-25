import torch

# -----------------------------
# FILE PATHS
# -----------------------------
files = [
    r"G:\final_phase_2\outputs\afew_model.pt",
    r"G:\final_phase_2\outputs\ravdess_model.pt",
    r"G:\final_phase_2\outputs\crema-d_model.pt"
]

# -----------------------------
# LOAD AND PRINT RESULTS
# -----------------------------
print("\n====== MODEL RESULTS ======")

for file in files:
    print("\n------------------------------")
    print(f"Loading: {file}")

    data = torch.load(file)

    info = data.get("dataset_info", {})

    dataset = info.get("dataset", "Unknown")
    acc = info.get("accuracy", "N/A")
    f1 = info.get("f1_score", "N/A")

    print(f"Dataset   : {dataset}")
    print(f"Accuracy  : {acc}%")
    print(f"F1 Score  : {f1}")