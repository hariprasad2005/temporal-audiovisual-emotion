import torch
import torch.nn as nn
import yaml
import os
from typing import Dict

from utils.logger import setup_logger
from utils.device import setup_device, set_seed
from data.loaders import create_dataloaders
from models.model import AudioVisualEmotionModel

def train_audio_visual_temporal(dataset_name: str, config: Dict):
    """Train audio-visual temporal model (proposed)"""
    logger = setup_logger(config['paths']['log_dir'], f'av_temporal_{dataset_name}')
    device = setup_device()
    set_seed(42)
    
    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        dataset_name, config, config['training']['batch_size']
    )
    
    # Create model
    model = AudioVisualEmotionModel(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for audio, frames, labels in train_loader:
            audio, frames, labels = audio.to(device), frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(audio, frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
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
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        logger.info(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | '
                   f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 
                      os.path.join(config['paths']['model_dir'], f'Audio_Visual_Temporal_{dataset_name}.pt'))
    
    logger.info(f'Best Accuracy: {best_accuracy:.2f}%')
    return best_accuracy

if __name__ == "__main__":
    config = yaml.safe_load(open('configs/config.yaml', 'r'))
    
    datasets = ['crema_d', 'ravdess', 'afew']
    results = {}
    
    for dataset in datasets:
        print(f"Training Audio-Visual Temporal model on {dataset}")
        acc = train_audio_visual_temporal(dataset, config)
        results[dataset] = acc
    
    print("Audio-Visual Temporal Results:")
    for dataset, acc in results.items():
        print(f"{dataset}: {acc:.2f}%")
