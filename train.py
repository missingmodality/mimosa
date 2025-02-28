import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from pathlib import Path

from model import MultiModalAutoencoder, compute_cindex, compute_time_dependent_concordance
from dataset import create_data_loaders, create_datalist
from pre import load_survival_data, normalize_survival_times, preprocess_survival_times
from config import ModelConfig

def setup_logging(config):
    """Configure logging with both file and console handlers"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'training_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return timestamp

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, timestamp, is_best=False):
    """Save model checkpoint with metrics"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if is_best:
        path = checkpoint_dir / f'best_model_{timestamp}.pt'
    else:
        path = checkpoint_dir / f'checkpoint_{timestamp}_epoch_{epoch}.pt'
    
    torch.save(checkpoint, path)
    logging.info(f'Saved {"best " if is_best else ""}checkpoint: {path}')

def validate(model, val_loader, device, epoch):
    """Run validation and compute metrics"""
    model.eval()
    val_metrics = {
        'val_loss': 0,
        'recon_loss': 0,
        'quant_loss': 0,
        'survival_loss': 0,
        'mi_loss': 0,
        'ortho_loss': 0,
        'contra_loss': 0
    }
    
    val_predictions = []
    val_true_times = []
    val_events = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch}'):
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            output = model(batch)
            loss_dict = model.compute_loss(batch, output)
            
            # Accumulate all losses
            for k, v in loss_dict.items():
                if k in val_metrics:
                    val_metrics[k] += v
            
            # Collect predictions for c-index and ctd-index
            val_predictions.append(output['survival_pred'].cpu())
            val_true_times.append(batch['rfs'].cpu())
            if 'event' in batch:
                val_events.append(batch['event'].cpu())
    
    # Average losses
    num_batches = len(val_loader)
    val_metrics = {k: v/num_batches for k, v in val_metrics.items()}
    
    # Calculate traditional c-index
    all_predictions = torch.cat(val_predictions, dim=0)
    all_true_times = torch.cat(val_true_times, dim=0)
    
    if len(val_events) > 0:
        all_events = torch.cat(val_events, dim=0)
        
        # DeepHit-specific: Calculate time-dependent concordance index for each event type
        # For survival_pred of shape [batch, num_events, num_time_bins]
        if len(all_predictions.shape) == 3:
            num_events = all_predictions.shape[1]
            ctd_indices = compute_time_dependent_concordance(
                all_predictions.numpy(), 
                all_true_times.numpy(), 
                all_events.numpy()
            )
            
            # Add each event's Ctd index to metrics
            for k in range(num_events):
                val_metrics[f'ctd_index_event_{k+1}'] = ctd_indices[k]
                
            # Average Ctd index across events
            val_metrics['ctd_index_avg'] = np.mean(ctd_indices)
        else:
            # Standard c-index for traditional single-output models
            val_metrics['c_index'] = compute_cindex(all_predictions.squeeze(), all_true_times)
    else:
        # Fallback to standard c-index if events are not available
        val_metrics['c_index'] = compute_cindex(all_predictions.squeeze(), all_true_times)
    
    return val_metrics

def train_epoch(model, train_loader, optimizer, scaler, device, epoch):
    """Run one epoch of training"""
    model.train()
    epoch_metrics = {
        'train_loss': 0,
        'recon_loss': 0,
        'quant_loss': 0,
        'survival_loss': 0,
        'mi_loss': 0,
        'ortho_loss': 0,
        'contra_loss': 0
    }
    
    train_predictions = []
    train_true_times = []
    train_events = []
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with automatic mixed precision
        with autocast(enabled=scaler is not None):
            output = model(batch)
            loss_dict = model.compute_loss(batch, output)
        
        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss_dict['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict['loss'].backward()
            optimizer.step()
        
        # Update metrics
        for k, v in loss_dict.items():
            if k in epoch_metrics:
                epoch_metrics[k] += v
        
        # Collect predictions
        train_predictions.append(output['survival_pred'].detach().cpu())
        train_true_times.append(batch['rfs'].cpu())
        if 'event' in batch:
            train_events.append(batch['event'].cpu())
        
        # Update progress bar with all losses
        progress_bar.set_postfix({
            'loss': loss_dict['loss'].item(),
            'recon': loss_dict['recon_loss'],
            'surv': loss_dict['survival_loss'],
            'mi': loss_dict['mi_loss']
        })
    
    # Average metrics
    num_batches = len(train_loader)
    epoch_metrics = {k: v/num_batches for k, v in epoch_metrics.items()}
    
    # Calculate training c-index or ctd-index
    all_predictions = torch.cat(train_predictions, dim=0)
    all_true_times = torch.cat(train_true_times, dim=0)
    
    if len(train_events) > 0:
        all_events = torch.cat(train_events, dim=0)
        
        # DeepHit-specific: Calculate time-dependent concordance index for each event type
        if len(all_predictions.shape) == 3:
            num_events = all_predictions.shape[1]
            ctd_indices = compute_time_dependent_concordance(
                all_predictions.numpy(), 
                all_true_times.numpy(), 
                all_events.numpy()
            )
            
            # Add each event's Ctd index to metrics
            for k in range(num_events):
                epoch_metrics[f'ctd_index_event_{k+1}'] = ctd_indices[k]
                
            # Average Ctd index across events
            epoch_metrics['ctd_index_avg'] = np.mean(ctd_indices)
        else:
            # Standard c-index for traditional models
            epoch_metrics['c_index'] = compute_cindex(all_predictions.squeeze(), all_true_times)
    else:
        # Fallback to standard c-index if events are not available
        epoch_metrics['c_index'] = compute_cindex(all_predictions.squeeze(), all_true_times)
    
    return epoch_metrics

def train(config):
    """Main training function"""
    # Setup
    timestamp = setup_logging(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Load and normalize survival data
    logging.info('Loading survival data...')
    survival_df = load_survival_data(os.path.join(config.base_dir, 'train.xlsx'))
    survival_df, min_time, max_time = normalize_survival_times(survival_df)
    
    # DeepHit: Preprocess survival times into discrete time bins
    if hasattr(config, 'num_time_bins') and config.num_time_bins > 0:
        survival_df = preprocess_survival_times(survival_df, max_time, config.num_time_bins)
        logging.info(f'Discretized survival times into {config.num_time_bins} time bins')
    
    # Create datalist and split into train/val
    datalist = create_datalist(config)
    split_idx = int(len(datalist) * config.train_val_split)
    train_list = datalist[:split_idx]
    val_list = datalist[split_idx:]
    
    # Create data loaders
    train_loader = create_data_loaders(train_list, survival_df, config, is_train=True)
    val_loader = create_data_loaders(val_list, survival_df, config, is_train=False)
    logging.info(f'Created data loaders: {len(train_loader)} training, {len(val_loader)} validation batches')

    # Initialize model and training components
    model = MultiModalAutoencoder(config).to(device)
    optimizer = model.configure_optimizers(config.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.lr_decay_factor,
        patience=config.lr_decay_patience, 
        verbose=True
    )
    scaler = GradScaler() if config.use_mixed_precision else None

    # Training loop
    best_val_loss = float('inf')
    best_val_cindex = 0
    best_val_ctd_index = 0
    early_stopping_counter = 0
    early_stopping_patience = 20
    
    logging.info('Starting training...')
    for epoch in range(config.epochs):
        # Training phase
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        
        # Validation phase
        val_metrics = validate(model, val_loader, device, epoch)
        
        # Update learning rate
        scheduler.step(val_metrics['val_loss'])
        
        # Logging
        log_message = f"\nEpoch {epoch+1}/{config.epochs}\n"
        log_message += f"{'Metric':<20} {'Training':<15} {'Validation':<15}\n"
        log_message += "-" * 50 + "\n"
        
        # Log all metrics
        all_metrics = set(list(train_metrics.keys()) + list(val_metrics.keys()))
        for metric in all_metrics:
            train_value = train_metrics.get(metric, float('nan'))
            val_value = val_metrics.get(metric, float('nan'))
            
            # Format metrics for display
            if isinstance(train_value, float) and isinstance(val_value, float):
                log_message += f"{metric:<20} {train_value:>15.6f} {val_value:>15.6f}\n"
            else:
                log_message += f"{metric:<20} {train_value!s:>15} {val_value!s:>15}\n"
        
        logging.info(log_message)
        
        # Save checkpoints and handle early stopping
        is_best_loss = val_metrics['val_loss'] < best_val_loss
        if is_best_loss:
            best_val_loss = val_metrics['val_loss']
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Track the best c-index or ctd-index
        if 'ctd_index_avg' in val_metrics and val_metrics['ctd_index_avg'] > best_val_ctd_index:
            best_val_ctd_index = val_metrics['ctd_index_avg']
            is_best = True
        elif 'c_index' in val_metrics and val_metrics['c_index'] > best_val_cindex:
            best_val_cindex = val_metrics['c_index']
            is_best = True
        else:
            is_best = is_best_loss  # Fall back to using loss as the best metric
        
        if epoch % config.save_every == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {**train_metrics, **val_metrics},
                config, timestamp, is_best
            )
        
        # Early stopping check
        if early_stopping_counter >= early_stopping_patience:
            logging.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Final logging
    logging.info(f'\nTraining completed after {epoch + 1} epochs')
    logging.info(f'Best validation loss: {best_val_loss:.6f}')
    
    if 'ctd_index_avg' in val_metrics:
        logging.info(f'Best validation Ctd-index: {best_val_ctd_index:.6f}')
    else:
        logging.info(f'Best validation c-index: {best_val_cindex:.6f}')

if __name__ == '__main__':
    config = ModelConfig()
    train(config)