import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from quantise import NormEMAVectorQuantizer
import segmentation_models_pytorch_3d as smp

from pre import CTDecoder,CTEncoder,PETDecoder,PETEncoder

def compute_time_dependent_concordance(predictions, times, events):
    """
    Calculate time-dependent concordance index (Ctd) for survival predictions
    with competing risks.
    
    Args:
        predictions: numpy array of shape [batch_size, num_events, num_time_bins]
                    containing predicted cumulative incidence functions
        times: numpy array of shape [batch_size] with actual event/censoring times
        events: numpy array of shape [batch_size] with event indicators 
                (0 for censored, 1,2,... for events)
    
    Returns:
        numpy array of Ctd indices for each event type
    """
    import numpy as np
    
    batch_size, num_events, num_time_bins = predictions.shape
    ctd_indices = np.zeros(num_events)
    
    # For each event type
    for k in range(num_events):
        # Calculate the sum of the conditional CIF at each time for this event
        # This represents the predicted risk for event k
        risk_scores = predictions[:, k, :].sum(axis=1)
        
        concordant_pairs = 0
        total_pairs = 0
        
        # For each pair of patients
        for i in range(batch_size):
            # Only consider patients who experienced event k
            if events[i] != k + 1:  # Events are 1-indexed in the data
                continue
                
            for j in range(batch_size):
                # Find acceptable pairs: i experienced event k at time t
                # and j was still under observation at time t
                if times[i] < times[j]:
                    total_pairs += 1
                    
                    # Check if the model correctly assigned higher risk to i than j
                    if risk_scores[i] > risk_scores[j]:
                        concordant_pairs += 1
        
        # Calculate Ctd index for this event
        if total_pairs > 0:
            ctd_indices[k] = concordant_pairs / total_pairs
        else:
            ctd_indices[k] = float('nan')  # No valid pairs for this event
    
    return ctd_indices


def preprocess_survival_times(survival_df, max_time, num_bins):
    """
    Discretize survival times into time bins for DeepHit model
    
    Args:
        survival_df: DataFrame containing survival data with 'RFS' column
        max_time: Maximum survival time to consider (typically 1.0 for normalized data)
        num_bins: Number of time bins to discretize into
        
    Returns:
        DataFrame with added 'time_bin' column
    """
    # Scale time to [0, num_bins-1] and convert to integer bin index
    survival_df['time_bin'] = ((survival_df['RFS'] / max_time) * (num_bins - 1)).astype(int)
    
    # Ensure values are in the correct range
    survival_df['time_bin'] = survival_df['time_bin'].clip(0, num_bins-1)
    
    return survival_df
def deephit_loss(predictions, times, events, alpha=0.1, sigma=0.1):
    """
    DeepHit loss function combining log-likelihood and ranking loss
    
    Args:
        predictions: model output of shape [batch_size, num_events, num_time_bins]
        times: survival or censoring times of shape [batch_size]
        events: event indicators of shape [batch_size] (0 for censored, 1+ for events)
        alpha: weight for the ranking loss
        sigma: parameter for the ranking loss function
    
    Returns:
        total loss
    """
    batch_size = predictions.size(0)
    num_events = predictions.size(1)
    num_time_bins = predictions.size(2)
    
    # Convert times to time bin indices
    time_indices = torch.clamp(times.long(), 0, num_time_bins - 1)
    
    # Log-likelihood loss (L1)
    ll_loss = 0
    for i in range(batch_size):
        if events[i] > 0:  # Uncensored
            event_idx = events[i] - 1  # Convert to 0-indexed
            ll_loss -= torch.log(predictions[i, event_idx, time_indices[i]] + 1e-8)
        else:  # Censored
            # Probability of surviving past the censoring time
            survival_prob = 1.0
            for k in range(num_events):
                cdf = torch.sum(predictions[i, k, :time_indices[i]+1])
                survival_prob -= cdf
            ll_loss -= torch.log(torch.clamp(survival_prob, 1e-8, 1.0))
    
    # Ranking loss (L2)
    rank_loss = 0
    for k in range(num_events):
        # Calculate CIF for each sample at their event time
        cifs = torch.zeros(batch_size)
        for i in range(batch_size):
            cifs[i] = torch.sum(predictions[i, k, :time_indices[i]+1])
        
        # Find acceptable pairs
        for i in range(batch_size):
            for j in range(batch_size):
                if events[i] == k+1 and time_indices[i] < time_indices[j]:
                    # Ranking loss using exponential function
                    rank_loss += torch.exp(-(cifs[i] - cifs[j]) / sigma)
    
    total_loss = ll_loss + alpha * rank_loss
    return total_loss / batch_size

def simple_mutual_information_loss(shared, specific_ct, specific_pet):
    """
    Simple mutual information minimization through correlation
    
    Args:
    - shared: Shared representation
    - specific_ct: CT-specific representation
    - specific_pet: PET-specific representation
    
    Returns:
    - Mutual information loss that encourages independence
    """
    # Correlation between shared and specific representations
    shared_ct_corr = torch.abs(F.cosine_similarity(shared, specific_ct)).mean()
    shared_pet_corr = torch.abs(F.cosine_similarity(shared, specific_pet)).mean()
    
    # Correlation between specific representations
    specific_cross_corr = torch.abs(F.cosine_similarity(specific_ct, specific_pet)).mean()
    
    # Total MI loss (we want to minimize these correlations)
    mi_loss = shared_ct_corr + shared_pet_corr + specific_cross_corr
    
    return mi_loss

def cross_modal_contrastive_loss(features_ct, features_pet, temperature=0.1):
    """
    Cross-modal contrastive loss between CT and PET features
    
    Args:
    - features_ct: CT features (B, D)
    - features_pet: PET features (B, D)
    - temperature: temperature scaling parameter
    
    Returns:
    - Cross-modal contrastive loss
    """
    # Normalize features
    features_ct = F.normalize(features_ct, dim=1)
    features_pet = F.normalize(features_pet, dim=1)
    
    # Compute similarities
    similarity_matrix = torch.matmul(features_ct, features_pet.t())
    
    # Compute row-wise and column-wise softmax
    row_softmax = F.softmax(similarity_matrix / temperature, dim=1)
    col_softmax = F.softmax(similarity_matrix / temperature, dim=0)
    
    # Compute loss
    row_loss = -torch.log(torch.diag(row_softmax) + 1e-8)
    col_loss = -torch.log(torch.diag(col_softmax) + 1e-8)
    
    return (torch.mean(row_loss) + torch.mean(col_loss)) / 2

def intra_modal_contrastive_loss(features, temperature=0.1):
    """
    Intra-modal contrastive loss between different samples of the same modality
    
    Args:
    - features: Features from one modality (B, D)
    - temperature: Temperature scaling parameter
    
    Returns:
    - Intra-modal contrastive loss comparing different samples
    """
    # Normalize features
    features = F.normalize(features, dim=1)
    batch_size = features.shape[0]
    
    # Create augmented version (could be replaced with actual augmentations)
    # Here using a simple random perturbation as example
    features_aug = features + 0.1 * torch.randn_like(features)
    features_aug = F.normalize(features_aug, dim=1)
    
    # Compute similarity matrix between original and augmented
    similarity_matrix = torch.matmul(features, features_aug.T) / temperature
    
    # For each sample, its positive pair is the corresponding augmented version (diagonal)
    # and all other samples are negatives
    labels = torch.arange(batch_size).to(features.device)
    
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss

def compute_representation_losses(
    shared_features_ct, 
    shared_features_pet, 
    specific_features_ct, 
    specific_features_pet,
    mi_weight=0.5,
    intra_modal_weight=0.5,
    cross_modal_weight=0.5,
    temperature=0.1
):
    """
    Compute comprehensive representation learning losses
    
    Args:
    - shared_features_ct: Shared CT features
    - shared_features_pet: Shared PET features
    - specific_features_ct: CT-specific features
    - specific_features_pet: PET-specific features
    - mi_weight: Weight for mutual information loss
    - intra_modal_weight: Weight for intra-modal contrastive loss
    - cross_modal_weight: Weight for cross-modal contrastive loss
    - temperature: Temperature for contrastive losses
    
    Returns:
    - Dictionary of representation learning losses
    """
    # Mutual Information Loss
    mi_loss = simple_mutual_information_loss(
        shared_features_ct, 
        specific_features_ct, 
        specific_features_pet
    ) * mi_weight
    
    # Intra-modal Contrastive Losses
    intra_ct_loss = intra_modal_contrastive_loss(
        shared_features_ct, 
        specific_features_ct, 
        temperature
    )
    intra_pet_loss = intra_modal_contrastive_loss(
        shared_features_pet, 
        specific_features_pet, 
        temperature
    )
    
    # Cross-modal Contrastive Loss
    cross_modal_loss = cross_modal_contrastive_loss(
        shared_features_ct, 
        shared_features_pet, 
        temperature
    )
    
    # Combine losses
    total_intra_loss = (intra_ct_loss + intra_pet_loss) * intra_modal_weight
    total_contrastive_loss = cross_modal_loss * cross_modal_weight
    
    # Comprehensive loss
    total_representation_loss = (
        mi_loss + 
        total_intra_loss + 
        total_contrastive_loss
    )
    
    return {
        'total_representation_loss': total_representation_loss,
        'mi_loss': mi_loss,
        'intra_ct_loss': intra_ct_loss,
        'intra_pet_loss': intra_pet_loss,
        'cross_modal_loss': cross_modal_loss
    }




def normalize_survival_times(survival_df):
    """Normalize survival times to [0,1] range"""
    max_time = survival_df['RFS'].max()
    min_time = survival_df['RFS'].min()
    survival_df['RFS'] = (survival_df['RFS'] - min_time) / (max_time - min_time)
    return survival_df, min_time, max_time

def preprocess_survival_times(survival_df, max_time, num_bins):
    """
    Discretize survival times into time bins
    """
    # Scale time to [0, num_bins-1]
    survival_df['time_bin'] = ((survival_df['RFS'] / max_time) * (num_bins - 1)).astype(int)
    
    # Ensure values are in the correct range
    survival_df['time_bin'] = survival_df['time_bin'].clip(0, num_bins-1)
    
    return survival_df


def load_survival_data(file_path):
    """
    Load survival data from Excel file with column name mapping
    """
    logging.info(f"Attempting to load survival data from: {file_path}")
    
    if not os.path.exists(file_path):
        logging.error(f"File not found at path: {file_path}")
        available_files = os.listdir(os.path.dirname(file_path))
        logging.info(f"Available files in directory: {available_files}")
        raise FileNotFoundError(f"File not found at: {file_path}")
    
    try:
        # Read the Excel file
        logging.info("Reading Excel file with openpyxl...")
        survival_df = pd.read_excel(file_path, engine='openpyxl')
        
        # Log initial data info
        logging.info(f"Initial data shape: {survival_df.shape}")
        logging.info(f"Columns found: {survival_df.columns.tolist()}")
        
        # Column mapping
        column_mapping = {
            'PatientID': 'id',
            'RFS': 'RFS'
        }
        
        # Verify required source columns exist
        required_source_columns = ['PatientID', 'RFS']
        missing_columns = [col for col in required_source_columns if col not in survival_df.columns]
        
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            logging.info(f"Available columns: {survival_df.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Rename columns according to mapping
        survival_df = survival_df.rename(columns=column_mapping)
        
        # Data cleaning
        survival_df['id'] = survival_df['id'].astype(str).str.strip()
        survival_df['RFS'] = pd.to_numeric(survival_df['RFS'], errors='coerce')
        
        # Remove missing values
        initial_rows = len(survival_df)
        survival_df = survival_df.dropna(subset=['RFS'])
        if len(survival_df) < initial_rows:
            logging.warning(f"Removed {initial_rows - len(survival_df)} rows with missing RFS values")
        
        logging.info("\nProcessed survival data info:")
        logging.info(survival_df.info())
        logging.info("\nFirst few rows:")
        logging.info(survival_df.head())
        
        return survival_df
        
    except Exception as e:
        logging.error(f"Error reading Excel file: {str(e)}")
        logging.error(f"Full error traceback:", exc_info=True)
        raise



def compute_cindex(predictions, true_times):
    """
    Compute concordance index for survival predictions
    """
    # Convert tensors to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(true_times):
        true_times = true_times.cpu().numpy()
    
    # For survival prediction, lower predictions should correspond to longer survival times
    predictions = -predictions  # Negative because higher risk should mean lower survival time
    
    # Calculate c-index
    c_index = concordance_index(true_times, predictions, np.ones_like(true_times))
    return c_index
def orthogonality_loss(shared_repr, specific_repr_ct, specific_repr_pet):
    """Compute orthogonality loss between shared and specific representations"""
    ortho_loss_ct = torch.norm(torch.mm(shared_repr.T, specific_repr_ct))**2
    ortho_loss_pet = torch.norm(torch.mm(shared_repr.T, specific_repr_pet))**2
    return ortho_loss_ct + ortho_loss_pet

def correlation_loss(specific_repr_ct, specific_repr_pet):
    """Compute correlation loss between modality-specific representations"""
    return torch.norm(torch.mm(specific_repr_ct, specific_repr_pet.T))**2

class SharedEncoder(nn.Module):
    def __init__(self, input_dim=512):
        super(SharedEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        )
    
    def forward(self, x):
        return self.net(x)

class ModalitySpecificEncoder(nn.Module):
    def __init__(self, input_dim=512):
        super(ModalitySpecificEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        )
    
    def forward(self, x):
        return self.net(x)

class DeepHitSurvivalHead(nn.Module):
    """
    DeepHit survival prediction head that predicts the joint distribution
    of survival time and competing events
    
    Args:
        shared_dim: Dimension of shared representations
        specific_dim: Dimension of modality-specific representations
        num_events: Number of competing events (risks)
        num_time_bins: Number of discrete time bins
        hidden_dims: List of hidden dimensions for sub-networks
    """
    def __init__(self, shared_dim=256, specific_dim=256, num_events=1, 
                 num_time_bins=100, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.num_events = num_events
        self.num_time_bins = num_time_bins
        
        # Combine all features
        input_dim = 2 * shared_dim + 2 * specific_dim
        
        # Shared sub-network
        shared_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
        
        self.shared_net = nn.Sequential(*shared_layers)
        
        # Event-specific sub-networks
        self.event_nets = nn.ModuleList()
        for k in range(num_events):
            event_layers = [
                nn.Linear(hidden_dims[-1] + input_dim, hidden_dims[-1]),  # Residual connection
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[-1]),
                nn.Dropout(0.2),
                nn.Linear(hidden_dims[-1], num_time_bins)
            ]
            self.event_nets.append(nn.Sequential(*event_layers))
        
    def forward(self, shared_ct, shared_pet, specific_ct, specific_pet):
        """
        Forward pass through DeepHit survival head
        
        Args:
            shared_ct: Shared representation from CT
            shared_pet: Shared representation from PET
            specific_ct: CT-specific representation
            specific_pet: PET-specific representation
            
        Returns:
            Tensor of shape [batch_size, num_events, num_time_bins] with 
            probability mass function over time and events
        """
        batch_size = shared_ct.shape[0]
        
        # Concatenate all features
        combined_features = torch.cat([
            shared_ct, shared_pet, specific_ct, specific_pet
        ], dim=1)
        
        # Get shared representation
        shared_features = self.shared_net(combined_features)
        
        # Get event-specific outputs with residual connections
        event_outputs = []
        for k in range(self.num_events):
            # Add residual connection
            event_input = torch.cat([shared_features, combined_features], dim=1)
            event_output = self.event_nets[k](event_input)
            event_outputs.append(event_output.unsqueeze(1))
        
        # Concatenate event outputs
        output = torch.cat(event_outputs, dim=1)  # [batch_size, num_events, num_time_bins]
        
        # Apply softmax across all events and times to get proper joint distribution
        flat_output = output.view(batch_size, -1)
        flat_softmax = F.softmax(flat_output, dim=1)
        
        # Reshape back to [batch_size, num_events, num_time_bins]
        return flat_softmax.view(batch_size, self.num_events, self.num_time_bins)


def deephit_loss(predictions, times, events, alpha=0.1, sigma=0.1):
    """
    DeepHit loss function combining log-likelihood and ranking loss
    
    Args:
        predictions: model output of shape [batch_size, num_events, num_time_bins]
        times: survival or censoring times (discretized time bin indices) [batch_size]
        events: event indicators [batch_size] (0 for censored, 1+ for events)
        alpha: weight for the ranking loss
        sigma: parameter for the ranking loss function
    
    Returns:
        total loss
    """
    batch_size = predictions.size(0)
    num_events = predictions.size(1)
    num_time_bins = predictions.size(2)
    
    # L1: Log-likelihood loss
    ll_loss = 0
    for i in range(batch_size):
        if events[i] > 0:  # Uncensored
            event_idx = events[i] - 1  # Convert to 0-indexed
            time_idx = times[i]
            
            # Probability of event k at time t
            ll_loss -= torch.log(predictions[i, event_idx, time_idx] + 1e-8)
        else:  # Censored
            # Probability of surviving past the censoring time
            # This is 1 - sum of probabilities of any event happening before or at censoring time
            survival_prob = 1.0
            for k in range(num_events):
                # Sum probabilities up to censoring time for each event
                event_cdf = torch.sum(predictions[i, k, :times[i]+1])
                survival_prob -= event_cdf
            
            ll_loss -= torch.log(torch.clamp(survival_prob, 1e-8, 1.0))
    
    # L2: Ranking loss
    rank_loss = 0
    for k in range(num_events):
        # For each event, calculate cumulative incidence function (CIF)
        # CIF is cumulative sum of probabilities over time
        cif = torch.cumsum(predictions[:, k, :], dim=1)
        
        # Find acceptable pairs for this event
        for i in range(batch_size):
            if events[i] != k+1:  # Only consider patients who experienced event k
                continue
                
            time_i = times[i]
            
            for j in range(batch_size):
                if i == j:
                    continue
                    
                # j must have survived longer than i
                if time_i < times[j]:
                    # CIF at time_i for both patients
                    risk_i = cif[i, time_i]
                    risk_j = cif[j, time_i]
                    
                    # Ranking loss using exponential function
                    rank_loss += torch.exp(-(risk_i - risk_j) / sigma)
    
    # Combine losses
    total_loss = ll_loss + alpha * rank_loss
    return total_loss / batch_size

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantise import NormEMAVectorQuantizer

# Import the contrastive loss functions
from contrastive_loss import (
    info_nce_loss, 
    cross_modal_contrastive_loss, 
    intra_modal_contrastive_loss
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantise import NormEMAVectorQuantizer

class UNet3DFeatureExtractor(nn.Module):
    """Feature extractor using 3D U-Net with EfficientNet encoder"""
    def __init__(self, in_channels=1, feature_dim=512):
        super().__init__()
        # Initialize U-Net model
        self.unet_model = smp.Unet(
            encoder_name="efficientnet-b0",
            in_channels=in_channels,
            classes=3,  # Doesn't matter as we'll extract features before the final layer
        )
        
        # We'll get features from the bottleneck layer
        self.feature_dim = feature_dim
        
        # Feature projection layer to get desired dimensionality
        bottleneck_dim = 1024  # This depends on the U-Net architecture
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, feature_dim)
        )
    
    def forward(self, x):
        """
        Extract features from a 3D volume using U-Net encoder
        
        Args:
            x: 3D volume tensor of shape [B, C, D, H, W]
            
        Returns:
            features: Tensor of shape [B, feature_dim]
        """
        # Get U-Net encoder features
        features = self.unet_model.encoder(x)
        
        # Use bottleneck (deepest) features
        bottleneck_features = features[-1]
        
        # Project to desired dimension
        projected_features = self.projection(bottleneck_features)
        
        return projected_features


class MultiModalAutoencoderWithUNet3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 3D U-Net feature extractors
        self.ct_extractor = UNet3DFeatureExtractor(in_channels=1, feature_dim=512)
        self.pet_extractor = UNet3DFeatureExtractor(in_channels=1, feature_dim=512)
        
        # CT projection heads
        self.ct_shared_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        )
        self.ct_specific_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        )
        
        # PET projection heads
        self.pet_shared_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        )
        self.pet_specific_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        )
        
        # Learned fusion layer for combining modalities
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        # Quantizers
        self.shared_quantizer = NormEMAVectorQuantizer(
            n_embed=config.ct_codebook_size,
            embedding_dim=256
        )
        self.ct_specific_quantizer = NormEMAVectorQuantizer(
            n_embed=config.ct_codebook_size,
            embedding_dim=256
        )
        self.pet_specific_quantizer = NormEMAVectorQuantizer(
            n_embed=config.pet_codebook_size,
            embedding_dim=256
        )
        
        # Decoders (remaining the same as your original architecture)
        self.ct_decoder = CTDecoder()
        self.pet_decoder = PETDecoder()
        
        # Survival prediction head
        self.survival_head = DeepHitSurvivalHead(
        feature_dim=256*2 + 256*2,  # Shared + specific features
        num_time_bins=config.num_time_bins,  # Number of time intervals
        num_events=config.num_events  # Number of competing events
        )
    
    # Add time discretization attributes for survival analysis
        self.max_time = config.max_survival_time
        self.num_time_bins = config.num_time_bins

        
        # Loss weights
        self.recon_weight = 1.0
        self.quant_weight = 0.1
        self.survival_weight = 1.0
        self.mi_weight = 0.1
        self.ortho_weight = 0.1
        self.contra_weight = 0.1
        self.corr_weight = 0.1
        self.intra_contra_weight = 0.1
        self.cross_contra_weight = 0.1
        self.temperature = 0.1  # Temperature for contrastive learning

    def forward(self, batch):
        # Extract features using 3D U-Net encoders
        ct_features = self.ct_extractor(batch['ct'])
        pet_features = self.pet_extractor(batch['pet'])
        
        # Project to shared and specific spaces
        ct_shared_raw = self.ct_shared_mlp(ct_features)
        ct_specific_raw = self.ct_specific_mlp(ct_features)
        
        pet_shared_raw = self.pet_shared_mlp(pet_features)
        pet_specific_raw = self.pet_specific_mlp(pet_features)
        
        # Create a single shared representation through learned fusion
        shared_features_raw = self.fusion_layer(torch.cat([ct_shared_raw, pet_shared_raw], dim=1))
        
        # Quantize the features
        shared_quant = self.shared_quantizer(
            shared_features_raw.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        ct_specific_quant = self.ct_specific_quantizer(
            ct_specific_raw.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        pet_specific_quant = self.pet_specific_quantizer(
            pet_specific_raw.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        
        # Get quantized features
        shared_features = shared_quant['embeddings'].squeeze(-1).squeeze(-1).squeeze(-1)
        ct_specific_features = ct_specific_quant['embeddings'].squeeze(-1).squeeze(-1).squeeze(-1)
        pet_specific_features = pet_specific_quant['embeddings'].squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Combine features for reconstruction
        ct_combined = shared_features + ct_specific_features
        pet_combined = shared_features + pet_specific_features
        
        # Reconstruct
        ct_recon = self.ct_decoder(ct_combined.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        pet_recon = self.pet_decoder(pet_combined.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        
        # Survival prediction
        survival_pred = self.survival_head(
            shared_features, shared_features,
            ct_specific_features, pet_specific_features
        )
        
        return {
            'ct_reconstruction': ct_recon,
            'pet_reconstruction': pet_recon,
            'shared_quant_loss': shared_quant['commitment_loss'],
            'ct_specific_quant_loss': ct_specific_quant['commitment_loss'],
            'pet_specific_quant_loss': pet_specific_quant['commitment_loss'],
            'shared_features': shared_features,
            'ct_specific_features': ct_specific_features,
            'pet_specific_features': pet_specific_features,
            'ct_shared_raw': ct_shared_raw,
            'pet_shared_raw': pet_shared_raw,
            'ct_specific_raw': ct_specific_raw,
            'pet_specific_raw': pet_specific_raw,
            'survival_pred': survival_pred
        }

    def compute_loss(self, batch, output):
        # 1. Reconstruction losses
        recon_loss = (
            F.mse_loss(output['ct_reconstruction'], batch['ct']) +
            F.mse_loss(output['pet_reconstruction'], batch['pet'])
        ) * self.recon_weight
        
        # 2. Quantization losses
        quant_loss = (
            output['shared_quant_loss'] +
            output['ct_specific_quant_loss'] + 
            output['pet_specific_quant_loss']
        ) * self.quant_weight
        
        # 3. Survival prediction loss
        survival_loss = deephit_loss(
            output['survival_pred'],
            batch['survival_time'],
            batch['event'],
            alpha=self.contra_weight,
            sigma=self.temperature
        )   
        
        # Get features for constraint losses
        S_CT = output['ct_shared_raw']  # CT shared features
        S_PET = output['pet_shared_raw']  # PET shared features
        P_CT = output['ct_specific_raw']  # CT specific features
        P_PET = output['pet_specific_raw']  # PET specific features
        
        # Combined shared features through learned fusion 
        S = self.fusion_layer(torch.cat([S_CT, S_PET], dim=1))
        
        # 4. Orthogonality Loss: L_ortho = ||S^T × P_CT||² + ||S^T × P_PET||²
        S_normalized = F.normalize(S, p=2, dim=1)
        P_CT_normalized = F.normalize(P_CT, p=2, dim=1)
        P_PET_normalized = F.normalize(P_PET, p=2, dim=1)
        
        ortho_loss_ct = torch.norm(torch.mm(S_normalized.T, P_CT_normalized))**2
        ortho_loss_pet = torch.norm(torch.mm(S_normalized.T, P_PET_normalized))**2
        ortho_loss = (ortho_loss_ct + ortho_loss_pet) * self.ortho_weight
        
        # 5. Mutual Information Minimization: L_MI = MI(S, P_CT) + MI(S, P_PET)
        # Using cosine similarity as a proxy for mutual information
        mi_loss_ct = torch.abs(F.cosine_similarity(S, P_CT, dim=1)).mean()
        mi_loss_pet = torch.abs(F.cosine_similarity(S, P_PET, dim=1)).mean()
        mi_loss = (mi_loss_ct + mi_loss_pet) * self.mi_weight
        
        # 6. Correlation Loss: L_corr = ||P_CT × P_PET^T||²
        # This discourages correlation between modality-specific features
        corr_loss = torch.norm(torch.mm(P_CT_normalized, P_PET_normalized.T))**2 * self.corr_weight
        
        # 7. Intra-modal Contrastive Loss
        def compute_intra_modal_loss(shared, specific, temp=self.temperature):
            # Normalize features
            shared_norm = F.normalize(shared, dim=1)
            specific_norm = F.normalize(specific, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(shared_norm, specific_norm.T) / temp
            
            # Positive pairs are along the diagonal
            positives = torch.diag(similarity_matrix)
            
            # Compute loss: -log(exp(sim(S, P_m)/τ) / Σ_neg exp(sim(S, P_neg)/τ))
            logits = torch.exp(similarity_matrix)
            log_prob = positives - torch.log(logits.sum(dim=1))
            
            return -log_prob.mean()
        
        intra_ct_loss = intra_modal_contrastive_loss(output['ct_specific_features'], 
                                                self.temperature)
        intra_pet_loss = intra_modal_contrastive_loss(output['pet_specific_features'], 
                                                 self.temperature)
        intra_contra_loss = (intra_ct_loss + intra_pet_loss) * self.intra_contra_weight
    
        
        # 8. Cross-modal Contrastive Loss
        def compute_cross_modal_loss(s_ct, s_pet, temp=self.temperature):
            # Normalize features
            s_ct_norm = F.normalize(s_ct, dim=1)
            s_pet_norm = F.normalize(s_pet, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(s_ct_norm, s_pet_norm.T) / temp
            
            # Positive pairs are along the diagonal
            positives = torch.diag(similarity_matrix)
            
            # Compute loss: -log(exp(sim(S_CT, S_PET)/τ) / Σ_neg exp(sim(S_i, S_j)/τ))
            logits = torch.exp(similarity_matrix)
            log_prob = positives - torch.log(logits.sum(dim=1))
            
            return -log_prob.mean()
        
        cross_contra_loss = compute_cross_modal_loss(S_CT, S_PET) * self.cross_contra_weight
        
        # Total loss
        total_loss = (
            recon_loss + 
            quant_loss + 
            survival_loss + 
            ortho_loss + 
            mi_loss + 
            corr_loss + 
            intra_contra_loss + 
            cross_contra_loss
        )
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss.item(),
            'quant_loss': quant_loss.item(),
            'survival_loss': survival_loss.item(),
            'ortho_loss': ortho_loss.item(),
            'mi_loss': mi_loss.item(),
            'corr_loss': corr_loss.item(),
            'intra_contra_loss': intra_contra_loss.item(),
            'cross_contra_loss': cross_contra_loss.item()
        }

    def configure_optimizers(self, learning_rate):
        return torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )