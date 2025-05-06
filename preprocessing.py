import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import AUGMENTATION_INTENSITY, IMG_SIZE


def get_preprocessing_transforms(mode='train'):
    """
    Get preprocessing transforms based on the mode and configured intensity
    
    Args:
        mode: 'train' or 'valid' or 'test'
    
    Returns:
        Appropriate preprocessing transformation pipeline
    """
    if mode == 'train':
        if AUGMENTATION_INTENSITY == 'none':
            return get_base_transforms()
        elif AUGMENTATION_INTENSITY == 'light':
            return get_light_augmentation()
        elif AUGMENTATION_INTENSITY == 'medium':
            return get_medium_augmentation()
        else:
            raise ValueError(f"Unknown augmentation intensity: {AUGMENTATION_INTENSITY}")
    else:
        return get_base_transforms()


def get_base_transforms():
    """Base transforms applied to all images"""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_light_augmentation():
    """Light augmentation appropriate for medical imaging"""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.RandomRotate90(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.95, 1.05),
            rotate=(-15, 15),
            fit_output=False,
            p=0.3
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
        ], p=0.2),
        A.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=0.1, 
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_medium_augmentation():
    """Medium augmentation - still conservative for medical imaging"""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.1, 
            rotate_limit=30, 
            border_mode=0, 
            p=0.5
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MedianBlur(blur_limit=5, p=0.5),
        ], p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, 
            contrast_limit=0.15, 
            p=0.5
        ),
        A.CoarseDropout(
            max_holes=8, 
            max_height=8, 
            max_width=8, 
            min_holes=2, 
            min_height=4, 
            min_width=4, 
            p=0.2
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def apply_smote(X, y):
    """
    Apply SMOTE to balance the classes
    
    Args:
        X: Features
        y: Labels
    
    Returns:
        X_resampled, y_resampled: Balanced data
    """
    from imblearn.over_sampling import SMOTE
    from config import SMOTE_SAMPLING_STRATEGY, SEED
    
    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=SEED,
        k_neighbors=5
    )
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled