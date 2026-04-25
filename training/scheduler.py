from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_scheduler(optimizer, config: Dict):
    if config['training'].get('scheduler', 'plateau') == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return scheduler
    return None
