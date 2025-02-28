import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import necessary modules from your project
from config import ModelConfig
from dataset import create_datalist, create_data_loaders
from model import MultiModalAutoencoder, compute_cindex, compute_time_dependent_concordance
from pre import load_survival_data, normalize_survival_times, preprocess_survival_times

def setup_logging(output_dir):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'testing_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )
    return timestamp

def predict_with_missing_modality(model, batch, device):
    """
    Predict survival when one modality is missing
    
    Args:
    - model: Trained MultiModalAutoencoder
    - batch: Dictionary containing available modalities
    - device: Torch device
    
    Returns:
    - Survival prediction
    """
    # Determine available modalities
    available_modalities = []
    if 'ct' in batch and batch['ct'] is not None:
        available_modalities.append('ct')
    if 'pet' in batch and batch['pet'] is not None:
        available_modalities.append('pet')
    
    if len(available_modalities) == 0:
        raise ValueError("No modalities available for prediction")
    
    # Move batch to device and handle potential None values
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    
    # For DeepHit model, we need to handle the format differently based on model architecture
    model.eval()
    with torch.no_grad():
        if len(available_modalities) == 2:
            # If both modalities are available, use the normal forward path
            output = model(batch)
            return output['survival_pred']
        
        # When only one modality is available
        # This implementation assumes your model has components that can handle single modality
        # You might need to adapt this based on your exact model architecture
        if hasattr(model, 'single_modality_forward'):
            # If the model has a specific method for single modality
            return model.single_modality_forward(batch, available_modalities[0])
        else:
            # Fallback implementation - create dummy data for missing modality
            dummy_batch = batch.copy()
            missing_modality = 'pet' if available_modalities[0] == 'ct' else 'ct'
            present_modality = available_modalities[0]
            
            # Create a tensor of zeros with same shape as the available modality
            dummy_shape = batch[present_modality].shape
            dummy_tensor = torch.zeros(dummy_shape, device=device)
            
            # Add the dummy tensor for the missing modality
            dummy_batch[missing_modality] = dummy_tensor
            
            # Run the model with the augmented batch
            output = model(dummy_batch)
            return output['survival_pred']

def deephit_evaluation(model, test_loader, device, config=None):
    """
    Evaluate a DeepHit model on test data
    
    Args:
    - model: Trained model with DeepHit survival head
    - test_loader: DataLoader for test data
    - device: Torch device
    - config: Model configuration
    
    Returns:
    - Dictionary of evaluation metrics
    """
    model.eval()
    
    # Store predictions and true values
    all_predictions = []
    all_true_times = []
    all_events = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating model"):
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            output = model(batch)
            
            # Collect predictions and true values
            all_predictions.append(output['survival_pred'].cpu())
            all_true_times.append(batch['rfs'].cpu())
            
            # Collect event information if available
            if 'event' in batch:
                all_events.append(batch['event'].cpu())
                
            # Collect patient IDs for detailed analysis
            if 'patient_id' in batch:
                all_patient_ids.append(batch['patient_id'])
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    true_times = torch.cat(all_true_times, dim=0)
    
    # Initialize metrics dictionary
    metrics = {}
    
    # For DeepHit model with competing risks, evaluate time-dependent concordance
    if len(all_events) > 0:
        events = torch.cat(all_events, dim=0)
        
        # If predictions are for multiple events or time bins
        if len(predictions.shape) == 3:  # [batch, num_events, num_time_bins]
            num_events = predictions.shape[1]
            
            # Calculate time-dependent concordance index for each event
            ctd_indices = compute_time_dependent_concordance(
                predictions.numpy(),
                true_times.numpy(),
                events.numpy()
            )
            
            # Store Ctd index for each event
            for k in range(num_events):
                metrics[f'ctd_index_event_{k+1}'] = ctd_indices[k]
            
            # Average Ctd index
            metrics['ctd_index_avg'] = np.mean(ctd_indices)
            
            # For each event, calculate CIF and other metrics
            for k in range(num_events):
                # Get predictions for this event (cumulative incidence function)
                event_predictions = predictions[:, k, :].sum(dim=1)
                
                # Calculate conventional c-index (for comparison)
                event_mask = (events == k+1)
                if event_mask.sum() > 0:
                    metrics[f'c_index_event_{k+1}'] = compute_cindex(
                        event_predictions[event_mask],
                        true_times[event_mask]
                    )
        else:
            # For traditional single-output model
            metrics['c_index'] = compute_cindex(predictions.squeeze(), true_times)
    else:
        # If no event information, calculate standard c-index
        metrics['c_index'] = compute_cindex(predictions.squeeze(), true_times)
    
    # Calculate standard regression metrics
    if len(predictions.shape) <= 2:  # Only for regression outputs
        pred_np = predictions.squeeze().numpy()
        true_np = true_times.numpy()
        metrics['mse'] = mean_squared_error(true_np, pred_np)
        metrics['mae'] = mean_absolute_error(true_np, pred_np)
    
    return metrics

def comprehensive_evaluation(model, test_loader, device, config, original_min_time=None, original_max_time=None):
    """
    Comprehensive model evaluation with different modality scenarios
    
    Args:
    - model: Trained MultiModalAutoencoder
    - test_loader: DataLoader for test set
    - device: Torch device
    - config: Model configuration
    - original_min_time: Minimum survival time (for denormalization)
    - original_max_time: Maximum survival time (for denormalization)
    
    Returns:
    - Dictionary of evaluation metrics
    """
    model.eval()
    
    # Prepare storage for predictions and true values
    results = {
        'both_modalities': {'true': [], 'pred': [], 'events': [], 'patient_ids': []},
        'ct_only': {'true': [], 'pred': [], 'events': [], 'patient_ids': []},
        'pet_only': {'true': [], 'pred': [], 'events': [], 'patient_ids': []}
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating scenarios"):
            # Clone batch to create different scenarios
            scenarios = [
                # Both modalities
                {
                    'ct': batch['ct'], 
                    'pet': batch['pet'], 
                    'rfs': batch['rfs'],
                    'patient_id': batch['patient_id'],
                    'scenario': 'both_modalities'
                },
                # CT only
                {
                    'ct': batch['ct'], 
                    'pet': None, 
                    'rfs': batch['rfs'],
                    'patient_id': batch['patient_id'],
                    'scenario': 'ct_only'
                },
                # PET only
                {
                    'ct': None, 
                    'pet': batch['pet'], 
                    'rfs': batch['rfs'],
                    'patient_id': batch['patient_id'],
                    'scenario': 'pet_only'
                }
            ]
            
            # Add event information if available
            if 'event' in batch:
                for scenario in scenarios:
                    scenario['event'] = batch['event']
            
            for scenario in scenarios:
                scenario_name = scenario.pop('scenario')
                
                # Move scenario to device
                scenario_batch = {
                    k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in scenario.items()
                }
                
                # Predict survival
                try:
                    survival_pred = predict_with_missing_modality(model, scenario_batch, device)
                    
                    # Store results
                    results[scenario_name]['true'].append(scenario_batch['rfs'].cpu())
                    results[scenario_name]['pred'].append(survival_pred.cpu())
                    
                    # Store events if available
                    if 'event' in scenario_batch:
                        results[scenario_name]['events'].append(scenario_batch['event'].cpu())
                    
                    # Store patient IDs
                    if 'patient_id' in scenario_batch:
                        results[scenario_name]['patient_ids'].append(scenario_batch['patient_id'])
                
                except Exception as e:
                    logging.error(f"Error in scenario {scenario_name}: {str(e)}")
    
    # Process and compute metrics for each scenario
    metrics = {}
    for scenario, data in results.items():
        if len(data['true']) == 0:
            continue
        
        # Concatenate data
        true_times = torch.cat(data['true'])
        pred_times = torch.cat(data['pred'])
        
        # Check if this is a DeepHit model with time bins
        if len(pred_times.shape) == 3:  # [batch, num_events, num_time_bins]
            # We have a DeepHit model
            if len(data['events']) > 0:
                events = torch.cat(data['events'])
                
                # Calculate time-dependent concordance
                num_events = pred_times.shape[1]
                ctd_indices = compute_time_dependent_concordance(
                    pred_times.numpy(),
                    true_times.numpy(),
                    events.numpy()
                )
                
                scenario_metrics = {
                    f'ctd_index_event_{k+1}': ctd_indices[k]
                    for k in range(num_events)
                }
                
                # Average Ctd index
                scenario_metrics['ctd_index_avg'] = np.mean(ctd_indices)
                
                # Add to overall metrics
                metrics[scenario] = scenario_metrics
            else:
                # If no event information, just use c-index on predicted risk
                # This is suboptimal for DeepHit models but provides some metric
                risk_score = pred_times.sum(dim=(1, 2))
                metrics[scenario] = {
                    'c_index': compute_cindex(risk_score, true_times)
                }
        else:
            # Traditional survival model with single output
            pred_flat = pred_times.squeeze()
            
            # Denormalize if original time range is provided
            if original_min_time is not None and original_max_time is not None:
                true_denorm = true_times * (original_max_time - original_min_time) + original_min_time
                pred_denorm = pred_flat * (original_max_time - original_min_time) + original_min_time
            else:
                true_denorm = true_times
                pred_denorm = pred_flat
            
            # Compute standard metrics
            metrics[scenario] = {
                'mse': mean_squared_error(true_denorm.numpy(), pred_denorm.numpy()),
                'mae': mean_absolute_error(true_denorm.numpy(), pred_denorm.numpy()),
                'c_index': compute_cindex(pred_flat, true_times)
            }
    
    return metrics

def test(config, checkpoint_path):
    """
    Main testing function
    
    Args:
    - config: Model configuration
    - checkpoint_path: Path to model checkpoint
    """
    # Setup logging
    timestamp = setup_logging(config.output_dir)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load survival data
    try:
        survival_df = load_survival_data(os.path.join(config.base_dir, 'train.xlsx'))
        
        # Normalize survival times and store original min/max for later denormalization
        normalized_df, original_min_time, original_max_time = normalize_survival_times(survival_df)
        
        # For DeepHit: Discretize survival times
        if hasattr(config, 'num_time_bins') and config.num_time_bins > 0:
            normalized_df = preprocess_survival_times(normalized_df, max_time=1.0, num_bins=config.num_time_bins)
            logging.info(f"Discretized survival times into {config.num_time_bins} time bins")
    except Exception as e:
        logging.error(f"Error loading survival data: {str(e)}")
        return
    
    # Create datalist
    datalist = create_datalist(config)
    
    # Split data
    _, test_datalist = train_test_split(
        datalist, 
        test_size=0.2, 
        random_state=42
    )
    
    # Create test data loader
    test_loader = create_data_loaders(
        test_datalist, 
        normalized_df,
        config, 
        is_train=False
    )
    
    # Initialize model
    model = MultiModalAutoencoder(config).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error loading checkpoint: {str(e)}")
        return
    
    # Check if this is a DeepHit model by inspecting the model architecture
    is_deephit = False
    if hasattr(model, 'survival_head') and hasattr(model.survival_head, 'num_time_bins'):
        is_deephit = True
        logging.info("Detected DeepHit survival model architecture")
    
    # Choose evaluation function based on model type
    if is_deephit:
        logging.info("Running DeepHit-specific evaluation")
        evaluation_results = deephit_evaluation(model, test_loader, device, config)
        
        # Log DeepHit results
        logging.info("\nDeepHit Evaluation Results:")
        for metric, value in evaluation_results.items():
            logging.info(f"{metric}: {value:.4f}" if isinstance(value, (float, np.float32, np.float64)) else f"{metric}: {value}")
    else:
        # Run comprehensive evaluation for traditional models
        logging.info("Running comprehensive evaluation with modality scenarios")
        evaluation_results = comprehensive_evaluation(
            model, 
            test_loader, 
            device, 
            config,
            original_min_time,
            original_max_time
        )
        
        # Log results
        logging.info("\nEvaluation Results:")
        for scenario, metrics in evaluation_results.items():
            logging.info(f"\nScenario: {scenario}")
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value:.4f}" if isinstance(value, (float, np.float32, np.float64)) else f"{metric}: {value}")
    
    # Save results to CSV
    results_df = pd.DataFrame()
    
    if is_deephit:
        # Format DeepHit results
        results_df = pd.DataFrame({
            'Metric': list(evaluation_results.keys()),
            'Value': list(evaluation_results.values())
        })
    else:
        # Format comprehensive evaluation results
        rows = []
        for scenario, metrics in evaluation_results.items():
            for metric, value in metrics.items():
                rows.append({
                    'Scenario': scenario,
                    'Metric': metric,
                    'Value': value
                })
        results_df = pd.DataFrame(rows)
    
    # Save to CSV
    results_path = os.path.join(config.output_dir, f'test_results_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    logging.info(f"Results saved to {results_path}")

if __name__ == '__main__':
    # Configuration and checkpoint setup
    config = ModelConfig()
    
    # Path to the best model checkpoint (adjust as needed)
    checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model_latest.pt')
    
    # Run testing
    test(config, checkpoint_path)