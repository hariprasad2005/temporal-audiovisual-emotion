import torch
import torchaudio
import torchvision.transforms as T
from torchvision.transforms.functional import resize
import librosa
import cv2
import numpy as np
from typing import List, Tuple
from facenet_pytorch import MTCNN

class AudioProcessor:
    def __init__(self, config: Dict):
        self.sample_rate = config['audio_sr']
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
    
    def process(self, audio_path: str) -> torch.Tensor:
        # Load and resample audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Normalize
        waveform = waveform / waveform.abs().max()
        
        # Extract features (using raw waveform for Wav2Vec2)
        return waveform.squeeze(0)

class VideoProcessor:
    def __init__(self, config: Dict):
        self.image_size = config['image_size']
        self.num_frames = config['num_frames']
        self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def extract_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Extract faces from frame using MTCNN"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(frame_rgb)
        
        faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame_rgb[y1:y2, x1:x2]
                if face.size > 0:
                    faces.append(face)
        
        return faces
    
    def process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Process a single frame"""
        faces = self.extract_faces(frame)
        if not faces:
            # Return blank frame if no faces detected
            blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            faces = [blank]
        
        # Use the first face detected
        face = faces[0]
        face = cv2.resize(face, (self.image_size, self.image_size))
        face = torch.from_numpy(face).float() / 255.0
        face = face.permute(2, 0, 1)  # HWC to CHW
        face = self.normalize(face)
        
        return face
    
    def process(self, video_path: str) -> torch.Tensor:
        """Process video and extract frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Uniform sampling
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                processed_frame = self.process_frame(frame)
                frames.append(processed_frame)
        
        cap.release()
        
        if len(frames) < self.num_frames:
            # Pad with last frame if needed
            last_frame = frames[-1] if frames else torch.zeros(3, self.image_size, self.image_size)
            while len(frames) < self.num_frames:
                frames.append(last_frame)
        
        return torch.stack(frames)  # [T, C, H, W]
