import os
import cv2
import pandas as pd
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from preprocessing import get_preprocessing_transforms, apply_smote
from config import (
    TRAIN_DIR, TEST_DIR, TRAIN_CSV, TEST_CSV, 
    BATCH_SIZE, NUM_FOLDS, SEED, APPLY_SMOTE, DEVICE
)

# Set up logging
logger = logging.getLogger('dataset')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class BoneFractureDataset(Dataset):
    """Dataset for bone fracture images with robust error handling"""
    
    def __init__(self, df, img_dir, transform=None, fallback_mode='skip'):
        """
        Args:
            df: DataFrame with image filenames and labels
            img_dir: Directory containing the images
            transform: Optional transforms to be applied
            fallback_mode: Strategy for handling missing/corrupted images:
                          'skip' - Skip invalid images during dataset creation
                          'blank' - Replace invalid images with blank images
        """
        self.img_dir = img_dir
        self.transform = transform
        self.fallback_mode = fallback_mode
        
        # Copy the DataFrame to avoid modifying the original
        self.df = df.copy()
        
        # Convert string labels to numeric if they're not already
        if 'label' in self.df.columns and isinstance(self.df['label'].iloc[0], str):
            label_map = {'fractured': 1, 'non_fractured': 0}
            self.df['label'] = self.df['label'].apply(lambda l: label_map.get(l.lower(), 0))
        
        # Validate images if fallback_mode is 'skip'
        if fallback_mode == 'skip':
            valid_indices = []
            invalid_files = []
            
            for idx, row in self.df.iterrows():
                img_path = os.path.join(self.img_dir, row['filename'])
                if os.path.exists(img_path):
                    # Try loading the image to check if it's valid
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            valid_indices.append(idx)
                        else:
                            invalid_files.append(row['filename'])
                    except Exception as e:
                        logger.warning(f"Error loading image {img_path}: {str(e)}")
                        invalid_files.append(row['filename'])
                else:
                    invalid_files.append(row['filename'])
            
            # Log info about invalid files
            if invalid_files:
                logger.warning(f"Found {len(invalid_files)} invalid/missing images out of {len(df)}")
                logger.warning(f"First 5 invalid files: {invalid_files[:5]}")
                
                # Keep only valid entries
                self.df = self.df.loc[valid_indices].reset_index(drop=True)
                logger.info(f"Dataset contains {len(self.df)} valid images after filtering")
        
        # Store filenames and labels as arrays for faster access
        self.filenames = self.df['filename'].values
        self.labels = self.df['label'].values if 'label' in self.df.columns else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            # Check if file exists
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            # Read image
            image = cv2.imread(img_path)
            
            # Check if image was loaded successfully
            if image is None:
                raise ValueError(f"Failed to decode image: {img_path}")
                
            # Convert color
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {str(e)}")
            
            if self.fallback_mode == 'blank':
                # Create a blank image (black) with appropriate dimensions
                # Use a standard size or get dimensions from your config
                image_size = 224  # Typical default size
                image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            else:
                # This shouldn't happen if using 'skip' mode, but just in case
                raise e
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        # Get label if available (for training/validation)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            # For test data without labels
            return image


# IMPORTANT: Moved SmoteDataset outside of the prepare_loaders function so it can be pickled
class SmoteDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Reshape back to image
        img_size = int(np.sqrt(feature.shape[0] // 3))
        image = feature.reshape(3, img_size, img_size).transpose(1, 2, 0)
        
        # Normalize to 0-255 range for albumentations
        image = (image * 255).astype(np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image, torch.tensor(label, dtype=torch.float32)


def prepare_loaders(fold=0):
    """
    Prepare training and validation data loaders for a specific fold
    
    Args:
        fold: Current fold number
    
    Returns:
        train_loader, val_loader, train_df, val_df
    """
    # Read data
    train_df = pd.read_csv(TRAIN_CSV)
    
    # Create folds
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    train_df['fold'] = -1
    
    for i, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        train_df.loc[val_idx, 'fold'] = i
    
    # Get current fold
    train_fold_df = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_fold_df = train_df[train_df['fold'] == fold].reset_index(drop=True)
    
    # Apply SMOTE to training data if configured
    if APPLY_SMOTE:
        try:
            # First, we load all training images to apply SMOTE
            logger.info("Initializing dataset for SMOTE with 'skip' mode to filter invalid images")
            temp_train_dataset = BoneFractureDataset(
                train_fold_df, 
                TRAIN_DIR, 
                transform=get_preprocessing_transforms('train'),
                fallback_mode='skip'  # Skip invalid images for SMOTE
            )
            
            # Create a simple loader to extract all features
            temp_loader = DataLoader(
                temp_train_dataset, 
                batch_size=64, 
                shuffle=False, 
                num_workers=0  # Use 0 workers for feature extraction to avoid potential issues
            )
            
            # Extract all features and labels
            all_features = []
            all_labels = []
            
            logger.info("Extracting features for SMOTE...")
            for images, labels in temp_loader:
                # Flatten the images for SMOTE
                batch_features = images.view(images.size(0), -1).cpu().numpy()
                all_features.append(batch_features)
                all_labels.append(labels.cpu().numpy())
            
            X = np.vstack(all_features)
            y = np.concatenate(all_labels)
            
            # Apply SMOTE
            logger.info("Applying SMOTE...")
            X_resampled, y_resampled = apply_smote(X, y)
            
            # Create datasets with SMOTE - using the globally defined SmoteDataset class
            logger.info(f"SMOTE applied - Original: {len(y)}, Resampled: {len(y_resampled)}")
            train_dataset = SmoteDataset(
                X_resampled, 
                y_resampled,
                transform=get_preprocessing_transforms('train')
            )
        except Exception as e:
            logger.error(f"SMOTE processing failed: {str(e)}")
            logger.info("Falling back to regular dataset without SMOTE")
            # Fallback to no SMOTE
            train_dataset = BoneFractureDataset(
                train_fold_df, 
                TRAIN_DIR, 
                transform=get_preprocessing_transforms('train'),
                fallback_mode='skip'
            )
    else:
        # Create datasets without SMOTE
        train_dataset = BoneFractureDataset(
            train_fold_df, 
            TRAIN_DIR, 
            transform=get_preprocessing_transforms('train'),
            fallback_mode='skip'
        )
    
    # Create validation dataset (never apply SMOTE to validation)
    val_dataset = BoneFractureDataset(
        val_fold_df, 
        TRAIN_DIR, 
        transform=get_preprocessing_transforms('valid'),
        fallback_mode='skip'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, # ALTERED: 4 -> 0
        pin_memory=True if DEVICE.type == "cuda" else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0, # ALTERED: 4 -> 0
        pin_memory=True if DEVICE.type == "cuda" else False
    )
    
    return train_loader, val_loader, train_fold_df, val_fold_df


def get_test_loader():
    """
    Prepare test data loader
    
    Returns:
        test_loader, test_df
    """
    test_df = pd.read_csv(TEST_CSV)
    
    test_dataset = BoneFractureDataset(
        test_df, 
        TEST_DIR, 
        transform=get_preprocessing_transforms('test'),
        fallback_mode='skip'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0, # ALTERED: 4 -> 0
        pin_memory=True if DEVICE.type == "cuda" else False
    )
    
    return test_loader, test_df


def validate_dataset(csv_path, img_dir):
    """
    Utility function to validate all images in a dataset and report issues
    
    Args:
        csv_path: Path to CSV file with image information
        img_dir: Directory containing the images
        
    Returns:
        DataFrame with validation results
    """
    df = pd.read_csv(csv_path)
    results = []
    
    logger.info(f"Validating {len(df)} images in {img_dir}...")
    
    for idx, row in df.iterrows():
        img_name = row['filename']
        img_path = os.path.join(img_dir, img_name)
        
        status = "valid"
        error_msg = None
        
        # Check if file exists
        if not os.path.exists(img_path):
            status = "missing"
            error_msg = "File not found"
        else:
            # Try to read the image
            try:
                img = cv2.imread(img_path)
                if img is None:
                    status = "corrupted"
                    error_msg = "Failed to decode image"
                else:
                    # Check image dimensions and channels
                    h, w, c = img.shape
                    if h == 0 or w == 0:
                        status = "invalid"
                        error_msg = f"Invalid dimensions: {h}x{w}"
            except Exception as e:
                status = "error"
                error_msg = str(e)
        
        results.append({
            'filename': img_name,
            'status': status,
            'error': error_msg
        })
        
        # Print progress
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} images")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary
    valid_count = (results_df['status'] == 'valid').sum()
    logger.info(f"Validation complete: {valid_count}/{len(df)} valid images")
    logger.info(f"Issues found: {len(df) - valid_count}")
    
    for status in ['missing', 'corrupted', 'invalid', 'error']:
        count = (results_df['status'] == status).sum()
        if count > 0:
            logger.info(f"  - {status}: {count}")
    
    return results_df