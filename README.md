# Temporal Facial Dynamics-Driven Audio-Visual Emotion Recognition

This repository implements a complete audio-visual emotion recognition system based on the research paper "Temporal Facial Dynamics-Driven Audio-Visual Emotion Recognition Using Vision Transformers". The system captures temporal facial dynamics across video sequences and fuses them with speech representations for robust emotion classification.

## Features

- **Temporal Modeling**: Captures facial expression evolution across multiple frames
- **Self-Supervised Backbones**: Uses Wav2Vec2 for audio and DINOv2 for visual features
- **Transformer Fusion**: Cross-modal attention for effective audio-visual integration
- **Cross-Dataset Evaluation**: Comprehensive evaluation across multiple datasets
- **Production Ready**: Modular, well-documented, and optimized for GPU training

## Supported Datasets

- CREMA-D
- RAVDESS  
- AFEW

## Emotion Classes

- happy
- sad
- angry
- neutral

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd emotion_recognition
