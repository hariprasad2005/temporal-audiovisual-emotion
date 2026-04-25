import torch
import yaml
import os
import pandas as pd
from typing import Dict
from evaluation.metrics import calculate_metrics

def evaluate_model_performance(model_type: str, dataset_name: str, config: Dict):
    """Evaluate specific model type on dataset"""
    from data.loaders import create_dataloaders
    from utils.device import get_device
    
    device = get_device()
    
    # Load model
    model_path = os.path.join(config['paths']['model_dir'], f'{model_type}_{dataset_name}.pt')
    
    if not os.path.exists(model_path):
        return None, None
    
    # Create appropriate model
    if model_type == 'Audio_Only':
        from train_audio_only import AudioOnlyModel
        model = AudioOnlyModel(config)
    elif model_type == 'Visual_Only':
        from train_visual_only import VisualOnlyModel
        model = VisualOnlyModel(config)
    elif model_type == 'Audio_Visual_Static':
        from train_audio_visual_static import AudioVisualStaticModel
        model = AudioVisualStaticModel(config)
    elif model_type == 'Audio_Visual_Temporal':
        from models.model import AudioVisualEmotionModel
        model = AudioVisualEmotionModel(config)
    else:
        return None, None
    
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Load data
    _, test_loader = create_dataloaders(dataset_name, config, config['training']['batch_size'])
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if model_type == 'Audio_Only':
                audio, _, labels = batch
                audio, labels = audio.to(device), labels.to(device)
                outputs = model(audio)
            elif model_type == 'Visual_Only':
                _, frames, labels = batch
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
            else:
                audio, frames, labels = batch
                audio, frames, labels = audio.to(device), frames.to(device), labels.to(device)
                outputs = model(audio, frames)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy, f1_score = calculate_metrics(all_preds, all_labels)
    return accuracy * 100, f1_score

def main():
    config = yaml.safe_load(open('configs/config.yaml', 'r'))
    datasets = ['crema_d', 'ravdess', 'afew']
    model_types = ['Audio_Only', 'Visual_Only', 'Audio_Visual_Static', 'Audio_Visual_Temporal']
    
    results = []
    
    print("Evaluating all models...")
    
    for model_type in model_types:
        for dataset in datasets:
            accuracy, f1 = evaluate_model_performance(model_type, dataset, config)
            if accuracy is not None:
                results.append({
                    'model_type': model_type,
                    'dataset': dataset,
                    'accuracy': accuracy,
                    'f1_score': f1
                })
                print(f"{model_type} on {dataset}: Acc={accuracy:.1f}%, F1={f1:.3f}")
    
    # Save results
    df = pd.DataFrame(results)
    results_path = os.path.join(config['paths']['output_dir'], 'all_model_results.csv')
    df.to_csv(results_path, index=False)
    
    # Create summary tables
    print("\n=== Final Results Summary ===")
    
    # Table I: Proposed model performance
    print("\nTable I: Proposed Audio-Visual Temporal Model")
    print("Dataset\t\tAccuracy\tF1-Score")
    print("-" * 40)
    temporal_results = df[df['model_type'] == 'Audio_Visual_Temporal']
    for _, row in temporal_results.iterrows():
        print(f"{row['dataset'].upper()}\t{row['accuracy']:.1f}%\t\t{row['f1_score']:.2f}")
    
    # Table III: Model comparison
    print("\nTable III: Model Performance Comparison")
    print("Model\t\t\tCREMA-D\tRAVDESS\tAFEW\tAvg Acc\tAvg F1")
    print("-" * 60)
    
    for model_type in model_types:
        model_results = df[df['model_type'] == model_type]
        if len(model_results) > 0:
            avg_acc = model_results['accuracy'].mean()
            avg_f1 = model_results['f1_score'].mean()
            crema = model_results[model_results['dataset'] == 'crema_d']['accuracy'].values[0]
            ravdess = model_results[model_results['dataset'] == 'ravdess']['accuracy'].values[0]
            afew = model_results[model_results['dataset'] == 'afew']['accuracy'].values[0]
            
            print(f"{model_type.replace('_', ' '):<20}{crema:.1f}\t{ravdess:.1f}\t{afew:.1f}\t{avg_acc:.1f}\t{avg_f1:.3f}")

if __name__ == "__main__":
    main()
