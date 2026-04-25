import torch

# -----------------------------
# FILE PATHS
# -----------------------------
files = [
    r"G:\final_phase_2\outputs\Audio_Only.pt",
    r"G:\final_phase_2\outputs\Visual.pt",
    r"G:\final_phase_2\outputs\Audio_Visual.pt"
]

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\nMODEL COMPARISON RESULTS\n")
print("{:<20} {:<10} {:<10}".format("Model", "Accuracy", "F1 Score"))

for file in files:
    data = torch.load(file)
    info = data["model_info"]

    print("{:<20} {:<10} {:<10}".format(
        info["model"],
        str(info["accuracy"]) + "%",
        info["f1_score"]
    ))