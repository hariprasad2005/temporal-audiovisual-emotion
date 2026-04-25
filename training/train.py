import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
from typing import Dict, List, Tuple
from ..utils.logger import setup_logger
from ..utils.device import get_device
from .trainer import EmotionTrainer
from .scheduler import get_scheduler

def train_model(model: nn.Module, train_loader, val_loader, config: Dict, 
                dataset_name: str, logger) -> Tuple[nn.Module, Dict]:
    
    device = get_device()
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = get_scheduler(optimizer, config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision
    scaler = GradScaler() if config['training']['mixed_precision'] else None
    
    # Trainer
    trainer = EmotionTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        scaler=scaler,
        device=device,
        config=config,
        logger=logger
    )
    
    # Train
    best_model, history = trainer.train(
        train_loader, val_loader, config['training']['num_epochs']
    )
    
    # Save model
    model_path = os.path.join(
        config['paths']['model_dir'], 
        f"{dataset_name}_model.pt"
    )
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'config': config,
        'history': history
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return best_model, history
