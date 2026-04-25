import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np
from .preprocessing import AudioProcessor, VideoProcessor
from .dataset_mapper import DatasetMapper

class AudioVisualDataset(Dataset):
    def __init__(self, dataset_name: str, data_path: str, config: Dict, 
                 split: str = "train", transform=None):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.config = config
        self.split = split
        self.transform = transform
        self.mapper = DatasetMapper(dataset_name)
        
        self.audio_processor = AudioProcessor(config)
        self.video_processor = VideoProcessor(config)
        
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        samples = []
        emotion_map = self.mapper.get_emotion_mapping()
        
        for emotion_dir in os.listdir(self.data_path):
            emotion_path = os.path.join(self.data_path, emotion_dir)
            if not os.path.isdir(emotion_path):
                continue
                
            emotion_label = emotion_map.get(emotion_dir.lower(), -1)
            if emotion_label == -1:
                continue
                
            for file in os.listdir(emotion_path):
                if file.endswith(self.config['audio_ext']):
                    base_name = os.path.splitext(file)[0]
                    video_file = base_name + self.config['video_ext']
                    video_path = os.path.join(emotion_path, video_file)
                    
                    if os.path.exists(video_path):
                        samples.append({
                            'audio_path': os.path.join(emotion_path, file),
                            'video_path': video_path,
                            'label': emotion_label
                        })
        
        # Split data (80% train, 20% test)
        np.random.seed(42)
        np.random.shuffle(samples)
        split_idx = int(len(samples) * 0.8)
        
        if self.split == "train":
            return samples[:split_idx]
        else:
            return samples[split_idx:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Process audio
        audio = self.audio_processor.process(sample['audio_path'])
        
        # Process video frames
        frames = self.video_processor.process(sample['video_path'])
        
        return audio, frames, sample['label']

def create_dataloaders(dataset_name: str, config: Dict, batch_size: int = 32):
    dataset_config = config['datasets'][dataset_name]
    
    train_dataset = AudioVisualDataset(
        dataset_name, dataset_config['path'], config, "train"
    )
    test_dataset = AudioVisualDataset(
        dataset_name, dataset_config['path'], config, "test"
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, test_loader
