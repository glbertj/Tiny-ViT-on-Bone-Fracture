import os
import torch
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# Create timestamp for unique run identification
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(BASE_DIR, "results", TIMESTAMP)
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "models")
PLOT_SAVE_PATH = os.path.join(RESULTS_DIR, "plots")
LOG_PATH = os.path.join(RESULTS_DIR, "logs")

# Create directories if they don't exist
for directory in [MODEL_SAVE_PATH, PLOT_SAVE_PATH, LOG_PATH]:
    os.makedirs(directory, exist_ok=True)

# Model configuration
MODEL_NAME = "vit_tiny_patch16_224"
IMG_SIZE = 224
CHANNELS = 3

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 30
NUM_FOLDS = 5
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
SWA_START = 10
SWA_LR = 1e-5
SEED = 42

# Class balancing
NUMBER_OF_CORES = "6"
APPLY_SMOTE = True
SMOTE_SAMPLING_STRATEGY = 'auto'
WEIGHTED_LOSS = True

os.environ["LOKY_MAX_CPU_COUNT"] = NUMBER_OF_CORES

# Augmentation intensity
# For medical images, we use conservative augmentations
AUGMENTATION_INTENSITY = 'light'  # Options: 'none', 'light', 'medium'

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
