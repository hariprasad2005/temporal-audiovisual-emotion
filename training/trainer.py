import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
from ..evaluation.metrics import calculate_metrics

class EmotionTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, scaler, device, config, logger):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.scaler = scaler
        self.device = device
        self.config = config
        self.logger = logger
        
        self.best_accuracy = 0.0
        self.best_model = None
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
    
    def train_epoch(self, train_loader) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training")
        for audio, frames, labels in pbar:
            audio = audio.to(self.device)
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(audio, frames)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                if self.config['training']['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip']
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(audio, frames)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip']
                    )
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate metrics
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy, f1_score = calculate_metrics(all_preds, all_labels)
        
        return avg_loss, accuracy, f1_score
    
    def validate(self, val_loader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for audio, frames, labels in val_loader:
                audio = audio.to(self.device)
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(audio, frames)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy, f1_score = calculate_metrics(all_preds, all_labels)
        
        return avg_loss, accuracy, f1_score
    
    def train(self, train_loader, val_loader, num_epochs: int) -> Tuple[nn.Module, Dict]:
        early_stopping_counter = 0
        patience = self.config['training']['early_stopping_patience']
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
            )
            
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.best_model = self.model.state_dict().copy()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model)
        
        return self.model, self.history
