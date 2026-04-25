import yaml
import os
from train_audio_only import train_audio_only
from train_visual_only import train_visual_only
from train_audio_visual_static import train_audio_visual_static
from train_audio_visual_temporal import train_audio_visual_temporal
from train_cross_dataset import main as train_cross_dataset

def main():
    config = yaml.safe_load(open('configs/config.yaml', 'r'))
    datasets = ['crema_d', 'ravdess', 'afew']
    
    print("Training all models as per research paper tables...")
    
    # Table III: Model Performance Comparison
    print("\n=== Training Audio-Only Models ===")
    audio_results = {}
    for dataset in datasets:
        acc = train_audio_only(dataset, config)
        audio_results[dataset] = acc
    
    print("\n=== Training Visual-Only Models ===")
    visual_results = {}
    for dataset in datasets:
        acc = train_visual_only(dataset, config)
        visual_results[dataset] = acc
    
    print("\n=== Training Audio-Visual Static Models ===")
    av_static_results = {}
    for dataset in datasets:
        acc = train_audio_visual_static(dataset, config)
        av_static_results[dataset] = acc
    
    print("\n=== Training Audio-Visual Temporal Models (Proposed) ===")
    av_temporal_results = {}
    for dataset in datasets:
        acc = train_audio_visual_temporal(dataset, config)
        av_temporal_results[dataset] = acc
    
    # Table I: Dataset-wise performance
    print("\n=== Dataset-wise Performance (Table I) ===")
    print("Dataset\t\tAccuracy\tF1-Score")
    print("-" * 40)
    for dataset in datasets:
        print(f"{dataset.upper()}\t{av_temporal_results[dataset]:.1f}%\t\t{av_temporal_results[dataset]/100:.2f}")
    
    # Table III: Model comparison
    print("\n=== Model Performance Comparison (Table III) ===")
    print("Model\t\t\tCREMA-D\tRAVDESS\tAFEW\tAvg")
    print("-" * 50)
    
    avg_audio = sum(audio_results.values()) / len(audio_results)
    avg_visual = sum(visual_results.values()) / len(visual_results)
    avg_av_static = sum(av_static_results.values()) / len(av_static_results)
    avg_av_temporal = sum(av_temporal_results.values()) / len(av_temporal_results)
    
    print(f"Audio-Only\t\t{audio_results['crema_d']:.1f}\t{audio_results['ravdess']:.1f}\t{audio_results['afew']:.1f}\t{avg_audio:.1f}")
    print(f"Visual-Only\t\t{visual_results['crema_d']:.1f}\t{visual_results['ravdess']:.1f}\t{visual_results['afew']:.1f}\t{avg_visual:.1f}")
    print(f"Audio-Visual Static\t{av_static_results['crema_d']:.1f}\t{av_static_results['ravdess']:.1f}\t{av_static_results['afew']:.1f}\t{avg_av_static:.1f}")
    print(f"Audio-Visual Temporal\t{av_temporal_results['crema_d']:.1f}\t{av_temporal_results['ravdess']:.1f}\t{av_temporal_results['afew']:.1f}\t{avg_av_temporal:.1f}")
    
    # Cross-dataset evaluation (Table II)
    print("\n=== Cross-Dataset Evaluation (Table II) ===")
    train_cross_dataset()

if __name__ == "__main__":
    main()
