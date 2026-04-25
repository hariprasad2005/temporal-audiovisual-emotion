import torch
import torch.nn as nn
import yaml
import os
import pandas as pd
from typing import Dict

from utils.logger import setup_logger
from utils.device import setup_device, set_seed
from data.loaders import create_dataloaders
from models.model import AudioVisualEmotionModel

def evaluate_cross_dataset(train_dataset: str, test_dataset: str, config: Dict):
    """Evaluate model trained on one dataset on another dataset"""
    logger = setup_logger(config['paths']['log_dir'], f'cross_{train_dataset}_to_{test_dataset}')
    device = setup_device()
    
    # Load model trained on source dataset
    model_path = os.path.join(config['paths']['model_dir'], f'Audio_Visual_Temporal_{train_dataset}.pt')
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return 0.0, 0.0
    
    model = AudioVisualEmotionModel(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load target dataset
    _, test_loader = create_dataloaders(
        test_dataset, config, config['training']['batch_size']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio, frames, labels in test_loader:
            audio, frames, labels = audio.to(device), frames.to(device), labels.to(device)
            outputs = model(audio, frames)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * correct / total
    val_loss = val_loss / len(test_loader)
    
    logger.info(f'{train_dataset} -> {test_dataset}: Acc: {val_acc:.2f}%, Loss: {val_loss:.4f}')
    
    # Save cross-dataset model
    cross_model_path = os.path.join(
        config['paths']['model_dir'], 
        f'{train_dataset}_to_{test_dataset}.pt'
    )
    torch.save(model.state_dict(), cross_model_path)
    
    return val_acc, val_loss

def main():
    config = yaml.safe_load(open('configs/config.yaml', 'r'))
    set_seed(42)
    
    datasets = ['crema_d', 'ravdess', 'afew']
    results = []
    
    # Cross-dataset evaluations as specified in Table II
    cross_pairs = [
        ('crema_d', 'ravdess'),
        ('crema_d', 'afew'),
        ('ravdess', 'crema_d')
    ]
    
    for train_dataset, test_dataset in cross_pairs:
        print(f"Evaluating {train_dataset} -> {test_dataset}")
        acc, loss = evaluate_cross_dataset(train_dataset, test_dataset, config)
        results.append({
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'accuracy': acc,
            'loss': loss
        })
    
    # Save results
    df = pd.DataFrame(results)
    results_path = os.path.join(config['paths']['output_dir'], 'cross_dataset_results.csv')
    df.to_csv(results_path, index=False)
    
    print("\nCross-Dataset Results:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
