###PRE_PROCESSING.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import os
from quantise import NormEMAVectorQuantizer
import numpy as np
from scipy.ndimage import zoom
from typing import Optional, Tuple
from lifelines.utils import concordance_index
import logging
import pandas as pd

# Add this function to normalize survival times
def normalize_survival_times(survival_df):
    """Normalize survival times to [0,1] range"""
    max_time = survival_df['RFS'].max()
    min_time = survival_df['RFS'].min()
    survival_df['RFS'] = (survival_df['RFS'] - min_time) / (max_time - min_time)
    return survival_df, min_time, max_time


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
def resize_to_match(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resize source array to match target array's dimensions"""
    if source.shape == target.shape:
        return source
        
    # Calculate zoom factors for each dimension
    factors = [t/s for s, t in zip(source.shape, target.shape)]
    
    # Resize using scipy's zoom
    resized = zoom(source, factors, order=1)
    return resized

def find_bounding_box(volume: np.ndarray) -> Tuple[slice, slice, slice]:
    """Find the 3D bounding box of non-zero elements in the volume."""
    coords = np.array(np.nonzero(volume))
    mins = np.min(coords, axis=1)
    maxs = np.max(coords, axis=1)
    return tuple(slice(min_, max_) for min_, max_ in zip(mins, maxs))

def center_crop_3d(volume: np.ndarray, target_size: int) -> np.ndarray:
    """Center crop a 3D volume to target size."""
    # Get current dimensions
    d, h, w = volume.shape
    
    # Calculate starting points for crop
    d_start = max(0, (d - target_size) // 2)
    h_start = max(0, (h - target_size) // 2)
    w_start = max(0, (w - target_size) // 2)
    
    # Crop
    cropped = volume[
        d_start:d_start + target_size,
        h_start:h_start + target_size,
        w_start:w_start + target_size
    ]
    
    # If the cropped volume is smaller than target_size, pad it
    if cropped.shape != (target_size, target_size, target_size):
        pad_depths = [(0, max(0, target_size - s)) for s in cropped.shape]
        cropped = np.pad(cropped, pad_depths, mode='constant')
    
    return cropped

def process_tumor_roi(volume: np.ndarray, 
                     mask: Optional[np.ndarray] = None,
                     target_size: int = 128) -> np.ndarray:
    """Process tumor ROI from a volume."""
    if mask is not None:
        # First resize mask to match volume dimensions if they don't match
        if mask.shape != volume.shape:
            mask = resize_to_match(mask, volume)
            
        # Ensure mask is binary
        mask = (mask > 0).astype(np.float32)
        
        # Apply mask to get tumor region
        tumor_region = volume * mask
        
        # Find bounding box
        bbox = find_bounding_box(mask)
        
        # Extract tumor region
        roi = tumor_region[bbox]
    else:
        roi = volume
    
    # Center crop to target size
    roi = center_crop_3d(roi, target_size)
    
    # Normalize to [-1, 1]
    if roi.max() != roi.min():
        roi = 2.0 * (roi - roi.min()) / (roi.max() - roi.min()) - 1.0
    
    return roi

###DATASET.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict
import os

def create_data_loaders(datalist, survival_df, config, is_train=True):
    """
    Create data loaders for training or validation
    
    Args:
        datalist: List of data samples
        survival_df: DataFrame containing survival data
        config: Configuration object
        is_train: Boolean indicating if this is for training
    """
    dataset = DualModalityDataset(datalist, survival_df, target_size=config.input_size)
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_train,
        num_workers=config.num_workers if is_train else 0,
        pin_memory=True
    )
    
    return loader

class DualModalityDataset(Dataset):
    def __init__(self, datalist: List[Dict], survival_df: pd.DataFrame, target_size: int = 128):
        self.datalist = datalist
        self.target_size = target_size
        # Set index to 'id' column after renaming
        self.survival_df = survival_df.set_index('id')
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        sample = self.datalist[idx]
        patient_id = sample['patient_id']
        
        # Load CT and PET volumes and mask
        ct_volume = np.load(sample['ct_path'])
        pet_volume = np.load(sample['pet_path'])
        mask = np.load(sample['mask_path'])
        
        # Process both modalities
        ct_processed = process_tumor_roi(ct_volume, mask, self.target_size)
        pet_processed = process_tumor_roi(pet_volume, mask, self.target_size)
        
        # Convert to tensors and add channel dimension
        ct_tensor = torch.from_numpy(ct_processed).float().unsqueeze(0)
        pet_tensor = torch.from_numpy(pet_processed).float().unsqueeze(0)
        
        # Get survival data using the renamed 'id' column
        try:
            rfs = self.survival_df.loc[patient_id, 'RFS']
            rfs_tensor = torch.tensor(rfs, dtype=torch.float32)
        except KeyError:
            logging.error(f"Patient ID {patient_id} not found in survival data")
            raise
        
        return {
            'ct': ct_tensor,
            'pet': pet_tensor,
            'rfs': rfs_tensor,
            'patient_id': patient_id
        }

def create_datalist(config):
    """Create list of data samples with paths for both CT and PET"""
    try:
        df = pd.read_csv(config.labels_file)
        print("\nDataFrame head:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())
        
    except Exception as e:
        print(f"Error reading labels file: {str(e)}")
        return []
    
    datalist = []
    for _, row in df.iterrows():
        patient_id = str(row['patient_id']).strip()
        
        sample = {
            'patient_id': patient_id,
            'ct_path': os.path.join(config.data_root, "images", f"{patient_id}.npy"),
            'pet_path': os.path.join(config.data_root, "pet_images", f"{patient_id}__PT.npy"),
            'mask_path': os.path.join(config.data_root, "masks", f"{patient_id}.npy"),
        }
        
        # Print first few samples for debugging
        if len(datalist) < 5:
            print(f"\nSample {len(datalist)+1}:")
            print(f"Patient ID: {patient_id}")
            print(f"CT path: {sample['ct_path']}")
            print(f"PET path: {sample['pet_path']}")
            print(f"Mask path: {sample['mask_path']}")
            print(f"Files exist: CT: {os.path.exists(sample['ct_path'])}, "
                  f"PET: {os.path.exists(sample['pet_path'])}, "
                  f"Mask: {os.path.exists(sample['mask_path'])}")
        
        # Only add if all files exist
        if (os.path.exists(sample['ct_path']) and 
            os.path.exists(sample['pet_path']) and 
            os.path.exists(sample['mask_path'])):
            datalist.append(sample)
    
    print(f"\nTotal samples found: {len(datalist)}")
    return datalist



###MODEL.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, stride=1, p=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=p),
            GroupNorm(ch_out),
            nn.SiLU(True)
        )
    
    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, ch, k_size=3):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=k_size, stride=1, padding=1),
            GroupNorm(ch),
            nn.SiLU(True),
            nn.Conv3d(ch, ch, kernel_size=k_size, stride=1, padding=1),
            GroupNorm(ch),
            nn.SiLU(True)
        )
    
    def forward(self, x):
        return x + self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(
                ch_in, 
                ch_out,
                kernel_size=2,
                stride=2,
                padding=0
            ),
            GroupNorm(ch_out),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.up(x)

class CTEncoder(nn.Module):
    def __init__(self):
        super(CTEncoder, self).__init__()
        self.conv1 = ConvBlock(ch_in=1, ch_out=32, k_size=4, stride=2)
        self.res1 = ResBlock(ch=32)
        
        self.conv2 = ConvBlock(ch_in=32, ch_out=64, k_size=4, stride=2)
        self.res2 = ResBlock(ch=64)
        
        self.conv3 = ConvBlock(ch_in=64, ch_out=128, k_size=4, stride=2)
        self.res3 = ResBlock(ch=128)
        
        self.conv4 = ConvBlock(ch_in=128, ch_out=256, k_size=4, stride=2)
        self.res4 = ResBlock(ch=256)
        
        self.conv5 = ConvBlock(ch_in=256, ch_out=512, k_size=3, stride=1)
        self.res5 = ResBlock(ch=512)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.res1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.res2(x2)
        
        x3 = self.conv3(x2)
        x3 = self.res3(x3)
        
        x4 = self.conv4(x3)
        x4 = self.res4(x4)
        
        x5 = self.conv5(x4)
        x5 = self.res5(x5)
        
        return x5

class PETEncoder(nn.Module):
    def __init__(self):
        super(PETEncoder, self).__init__()
        self.conv1 = ConvBlock(ch_in=1, ch_out=32, k_size=4, stride=2)
        self.res1 = ResBlock(ch=32)
        
        self.conv2 = ConvBlock(ch_in=32, ch_out=64, k_size=4, stride=2)
        self.res2 = ResBlock(ch=64)
        
        self.conv3 = ConvBlock(ch_in=64, ch_out=128, k_size=4, stride=2)
        self.res3 = ResBlock(ch=128)
        
        self.conv4 = ConvBlock(ch_in=128, ch_out=256, k_size=4, stride=2)
        self.res4 = ResBlock(ch=256)
        
        self.conv5 = ConvBlock(ch_in=256, ch_out=512, k_size=3, stride=1)
        self.res5 = ResBlock(ch=512)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.res1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.res2(x2)
        
        x3 = self.conv3(x2)
        x3 = self.res3(x3)
        
        x4 = self.conv4(x3)
        x4 = self.res4(x4)
        
        x5 = self.conv5(x4)
        x5 = self.res5(x5)
        
        return x5

class CTDecoder(nn.Module):
    def __init__(self):
        super(CTDecoder, self).__init__()
        self.up4 = UpBlock(512, 256)
        self.res4 = ResBlock(256)
        
        self.up3 = UpBlock(256, 128)
        self.res3 = ResBlock(128)
        
        self.up2 = UpBlock(128, 64)
        self.res2 = ResBlock(64)
        
        self.up1 = UpBlock(64, 32)
        self.res1 = ResBlock(32)
        
        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        if x.numel() > 2**31:
            raise ValueError(f"Input tensor too large: {x.shape}, {x.numel()} elements")
            
        x = self.up4(x)
        x = self.res4(x)
        
        x = self.up3(x)
        x = self.res3(x)
        
        x = self.up2(x)
        x = self.res2(x)
        
        x = self.up1(x)
        x = self.res1(x)
        
        x = self.final_conv(x)
        x = self.final_act(x)
        return x

class PETDecoder(nn.Module):
    def __init__(self):
        super(PETDecoder, self).__init__()
        self.up4 = UpBlock(512, 256)
        self.res4 = ResBlock(256)
        
        self.up3 = UpBlock(256, 128)
        self.res3 = ResBlock(128)
        
        self.up2 = UpBlock(128, 64)
        self.res2 = ResBlock(64)
        
        self.up1 = UpBlock(64, 32)
        self.res1 = ResBlock(32)
        
        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        if x.numel() > 2**31:
            raise ValueError(f"Input tensor too large: {x.shape}, {x.numel()} elements")
            
        x = self.up4(x)
        x = self.res4(x)
        
        x = self.up3(x)
        x = self.res3(x)
        
        x = self.up2(x)
        x = self.res2(x)
        
        x = self.up1(x)
        x = self.res1(x)
        
        x = self.final_conv(x)
        x = self.final_act(x)
        return x

# Modified Dataset class
class DualModalityDataset(Dataset):
    def __init__(self, datalist: List[Dict], survival_df: pd.DataFrame, target_size: int = 128):
        self.datalist = datalist
        self.target_size = target_size
        self.survival_df = survival_df.set_index('id')  # Index by patient ID

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        sample = self.datalist[idx]
        patient_id = sample['patient_id']
        
        # Load CT and PET volumes and mask
        ct_volume = np.load(sample['ct_path'])
        pet_volume = np.load(sample['pet_path'])
        mask = np.load(sample['mask_path'])
        
        # Process both modalities
        ct_processed = process_tumor_roi(ct_volume, mask, self.target_size)
        pet_processed = process_tumor_roi(pet_volume, mask, self.target_size)
        
        # Convert to tensors and add channel dimension
        ct_tensor = torch.from_numpy(ct_processed).float().unsqueeze(0)
        pet_tensor = torch.from_numpy(pet_processed).float().unsqueeze(0)
        
        # Get survival data
        rfs = self.survival_df.loc[patient_id, 'RFS']
        rfs_tensor = torch.tensor(rfs, dtype=torch.float32)
        
        return {
            'ct': ct_tensor, 
            'pet': pet_tensor,
            'rfs': rfs_tensor,
            'patient_id': patient_id
        }

# Survival Prediction Head
class SurvivalHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super(SurvivalHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),  # *2 because we concatenate CT and PET features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, ct_features, pet_features):
        # Global average pooling of features
        ct_features = F.adaptive_avg_pool3d(ct_features, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        pet_features = F.adaptive_avg_pool3d(pet_features, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Concatenate features
        combined_features = torch.cat([ct_features, pet_features], dim=1)
        
        # Predict survival
        return self.net(combined_features)

# Modified DualModalityAutoencoder
# Fixed DualModalityAutoencoder class
class DualModalityAutoencoder(nn.Module):
    def __init__(self, ct_codebook_size=1024, pet_codebook_size=1024):
        super(DualModalityAutoencoder, self).__init__()
        # Existing components
        self.ct_encoder = CTEncoder()
        self.pet_encoder = PETEncoder()
        self.ct_quantizer = NormEMAVectorQuantizer(n_embed=ct_codebook_size, embedding_dim=32)
        self.pet_quantizer = NormEMAVectorQuantizer(n_embed=pet_codebook_size, embedding_dim=32)
        self.ct_decoder = CTDecoder()
        self.pet_decoder = PETDecoder()
        
        # Add survival prediction head
        self.survival_head = SurvivalHead(512)  # 512 is the feature dimension from encoders
        # Adjust the weights to emphasize survival prediction
        self.recon_weight = 0.5  # Reduced from 1.0
        self.quant_weight = 0.1  # Reduced from 0.25
        self.survival_weight = 2.0  # Increased from 0.1
    
    def configure_optimizers(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    def forward(self, batch):
        # Process CT
        ct_features = self.ct_encoder(batch['ct'])
        ct_quant_output = self.ct_quantizer(ct_features)
        ct_quant_features = ct_quant_output['embeddings']  # Changed from 'quantize' to 'embeddings'
        ct_quant_loss = ct_quant_output['commitment_loss']
        ct_reconstruction = self.ct_decoder(ct_quant_features)
        
        # Process PET
        pet_features = self.pet_encoder(batch['pet'])
        pet_quant_output = self.pet_quantizer(pet_features)
        pet_quant_features = pet_quant_output['embeddings']  # Changed from 'quantize' to 'embeddings'
        pet_quant_loss = pet_quant_output['commitment_loss']
        pet_reconstruction = self.pet_decoder(pet_quant_features)
        
        # Survival prediction
        survival_pred = self.survival_head(ct_quant_features, pet_quant_features)
        
        return {
            'ct_reconstruction': ct_reconstruction,
            'pet_reconstruction': pet_reconstruction,
            'ct_quant_loss': ct_quant_loss * self.quant_weight,
            'pet_quant_loss': pet_quant_loss * self.quant_weight,
            'survival_pred': survival_pred
        }
    
    def compute_loss(self, batch, output):
        # Reconstruction losses (MSE)
        ct_recon_loss = F.mse_loss(output['ct_reconstruction'], batch['ct']) * self.recon_weight
        pet_recon_loss = F.mse_loss(output['pet_reconstruction'], batch['pet']) * self.recon_weight
        
        # Quantization losses (already weighted in forward pass)
        ct_quant_loss = output['ct_quant_loss']
        pet_quant_loss = output['pet_quant_loss']
        
        # Survival loss (MSE with weight)
        survival_loss = F.mse_loss(output['survival_pred'].squeeze(), batch['rfs']) * self.survival_weight
        
        # Total loss
        total_loss = ct_recon_loss + pet_recon_loss + ct_quant_loss + pet_quant_loss + survival_loss
        
        return {
            'loss': total_loss,
            'ct_recon_loss': ct_recon_loss.item(),
            'pet_recon_loss': pet_recon_loss.item(),
            'ct_quant_loss': ct_quant_loss.item(),
            'pet_quant_loss': pet_quant_loss.item(),
            'survival_loss': survival_loss.item()
        }
    
   


###CONFIG.py
class ModelConfig:
    def __init__(self):
        # Data Paths
        self.base_dir = "/"
        
        # Data Structure
        self.data_root = os.path.join(self.base_dir, "Train")
        self.ct_dir = os.path.join(self.data_root, "images")
        self.pet_dir = os.path.join(self.data_root, "pet_images")
        self.mask_dir = os.path.join(self.data_root, "masks")
        self.labels_file = os.path.join(self.data_root, "labels.csv")
        
        # Output directories
        self.output_dir = os.path.join(self.base_dir, "results")
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")
        
        # Model Architecture
        self.feature_dim = 512
        self.ct_codebook_size = 1024
        self.pet_codebook_size = 1024
        self.embedding_dim = 32
        
        # Training
        self.batch_size = 1
        self.epochs = 10000
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        
        # Data
        self.input_size = 128
        self.train_val_split = 0.8
        self.num_workers = 4
        
        # Loss weights
        self.ct_loss_weight = 1.0
        self.pet_loss_weight = 1.0
        
        # Optimization
        self.use_mixed_precision = True
        self.lr_decay = True
        self.lr_decay_factor = 0.5
        self.lr_decay_patience = 10
        
        # Logging
        self.save_every = 10
        self.validate_every = 1
        
        # Preprocessing
        self.normalize_range = (-1, 1)
        self.min_tumor_slices = 16

    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.data_root,
            self.ct_dir,
            self.pet_dir,
            self.mask_dir,
            self.output_dir,
            self.checkpoint_dir,
            self.log_dir
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created/verified directory: {directory}")

