import matplotlib.pyplot as plt
import os
from typing import Dict, List

def plot_training_history(history: Dict, save_dir: str, dataset_name: str):
    """Plot training history graphs"""
    # Loss plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    loss_path = os.path.join(save_dir, f"{dataset_name}_training_history.png")
    plt.savefig(loss_path)
    plt.close()
    
    # F1-score plot
    plt.figure(figsize=(6, 4))
    plt.plot(history['train_f1'], label='Train F1-Score')
    plt.plot(history['val_f1'], label='Validation F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.title('Training and Validation F1-Score')
    
    f1_path = os.path.join(save_dir, f"{dataset_name}_f1_history.png")
    plt.savefig(f1_path)
    plt.close()
