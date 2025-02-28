import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
from typing import List, Dict
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
from typing import List, Dict
from pre import process_tumor_roi, normalize_survival_times

from pre import process_tumor_roi, normalize_survival_times

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
        
        # Only add if all files exist
        if (os.path.exists(sample['ct_path']) and 
            os.path.exists(sample['pet_path']) and 
            os.path.exists(sample['mask_path'])):
            datalist.append(sample)
    
    print(f"\nTotal samples found: {len(datalist)}")
    return datalist