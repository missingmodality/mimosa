import os

class ModelConfig:
    def __init__(self):
        # Data Paths
        self.base_dir = "/path/to/data/seg/"
        
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
        self.epochs = 1000
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        
        # Data
        self.input_size = 128
        self.train_val_split = 0.8
        self.num_workers = 4
        
        # Loss weights
        self.ct_loss_weight = 1.0
        self.pet_loss_weight = 1.0
        self.survival_loss_weight = 1.0
        self.ortho_loss_weight = 0.1
        self.corr_loss_weight = 0.1
        
        # DeepHit Survival Parameters
        self.use_deephit = True  # Set to True to use DeepHit model, False for traditional regression
        self.num_events = 1  # Number of competing risks (1 for single-event survival)
        self.num_time_bins = 100  # Number of discrete time intervals
        self.deephit_hidden_dims = [512, 256, 128]  # Hidden dimensions for DeepHit network
        
        
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
        #self.min_tumor_slices = 16

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