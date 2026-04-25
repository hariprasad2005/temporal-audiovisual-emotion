import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import List, Tuple

def calculate_metrics(predictions: List[int], labels: List[int]) -> Tuple[float, float]:
    """Calculate accuracy and F1-score"""
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return accuracy, f1

def plot_confusion_matrix(labels: List[int], predictions: List[int], 
                         class_names: List[str], save_path: str):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_performance_report(results: dict, save_path: str):
    """Generate performance report in markdown format"""
    report = "# Emotion Recognition Performance Report\n\n"
    
    # Table I: Dataset-wise performance
    report += "## Table I: Dataset-wise Performance\n\n"
    report += "| Dataset | Accuracy | F1-Score |\n"
    report += "|---------|----------|----------|\n"
    for dataset, metrics in results.items():
        if 'accuracy' in metrics:
            report += f"| {dataset.upper()} | {metrics['accuracy']:.1f}% | {metrics['f1_score']:.2f} |\n"
    
    report += "\n## Table III: Model Comparison\n\n"
    report += "| Model | Accuracy | F1-Score |\n"
    report += "|-------|----------|----------|\n"
    
    # Add model comparison data
    model_types = ['Audio_Only', 'Visual_Only', 'Audio_Visual_Static', 'Audio_Visual_Temporal']
    for model_type in model_types:
        acc_values = []
        f1_values = []
        for dataset_metrics in results.values():
            if model_type in dataset_metrics:
                acc_values.append(dataset_metrics[model_type]['accuracy'])
                f1_values.append(dataset_metrics[model_type]['f1_score'])
        
        if acc_values:
            avg_acc = sum(acc_values) / len(acc_values)
            avg_f1 = sum(f1_values) / len(f1_values)
            report += f"| {model_type.replace('_', ' ')} | {avg_acc:.1f}% | {avg_f1:.2f} |\n"
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    return report
