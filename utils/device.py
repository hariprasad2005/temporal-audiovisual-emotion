import torch

def get_device() -> torch.device:
    """Get available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def setup_device():
    """Setup and log device information"""
    device = get_device()
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device
