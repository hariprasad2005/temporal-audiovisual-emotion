import argparse
import yaml
import os
from typing import Dict

from utils.logger import setup_logger
from utils.device import setup_device, set_seed
from utils.visualization import plot_training_history

from data.loaders import create_dataloaders
from models.model import AudioVisualEmotionModel
from training.train import train_model
from evaluation.evaluate import evaluate_model
from evaluation.cross_eval import CrossDatasetEvaluator

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Audio-Visual Emotion Recognition')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='Path to config file')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--eval', action='store_true', help='Evaluate models')
    parser.add_argument('--cross', action='store_true', help='Cross-dataset evaluation')
    parser.add_argument('--datasets', nargs='+', default=['crema_d', 'ravdess', 'afew'],
                       help='Datasets to process')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device and seed
    device = setup_device()
    set_seed(42)
    
    # Create output directories
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    os.makedirs(config['paths']['graph_dir'], exist_ok=True)
    
    # Setup logger
    logger = setup_logger(config['paths']['log_dir'], 'emotion_recognition')
    
    if args.train:
        logger.info("Starting training...")
        
        for dataset_name in args.datasets:
            logger.info(f"Training on {dataset_name}")
            
            # Create data loaders
            train_loader, test_loader = create_dataloaders(
                dataset_name, config, config['training']['batch_size']
            )
            
            # Create model
            model = AudioVisualEmotionModel(config)
            
            # Train model
            best_model, history = train_model(
                model, train_loader, test_loader, config, dataset_name, logger
            )
            
            # Plot training history
            plot_training_history(
                history, config['paths']['graph_dir'], dataset_name
            )
            
            # Evaluate on test set
            evaluate_model(
                best_model, test_loader, config, dataset_name, logger
            )
    
    if args.eval:
        logger.info("Starting evaluation...")
        
        for dataset_name in args.datasets:
            logger.info(f"Evaluating on {dataset_name}")
            
            # Load model
            model_path = os.path.join(
                config['paths']['model_dir'],
                f"{dataset_name}_model.pt"
            )
            
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                continue
            
            model = AudioVisualEmotionModel(config)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Create data loader
            _, test_loader = create_dataloaders(
                dataset_name, config, config['training']['batch_size']
            )
            
            # Evaluate
            evaluate_model(
                model, test_loader, config, dataset_name, logger
            )
    
    if args.cross:
        logger.info("Starting cross-dataset evaluation...")
        
        evaluator = CrossDatasetEvaluator(config, logger)
        evaluator.run_evaluation(args.datasets)

if __name__ == "__main__":
    main()
