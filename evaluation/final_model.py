import torch

files = [
    r"G:\final_phase_2\outputs\Audio_Visual_Temporal.pt"
]

print("\nMODEL COMPARISON RESULTS\n")
print("{:<30} {:<10} {:<10}".format("Model", "Accuracy", "F1 Score"))

for file in files:
    data = torch.load(file)
    info = data["model_info"]

    # Replace name here
    model_name = "Temporal Facial Dynamics"

    print("{:<30} {:<10} {:<10}".format(
        model_name,
        str(info["accuracy"]) + "%",
        info["f1_score"]
    ))