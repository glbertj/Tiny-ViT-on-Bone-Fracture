import os
import glob
import random
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import timm
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_DIR = "images"
MODEL_SAVE_PATH = "result/model/best_model.pth"
PLOT_SAVE_PATH = "result/plots"
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 20
NUM_FOLDS = 5
SWA_START = 5
LEARNING_RATE = 1e-4
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class FractureDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label

# Load image paths and labels
fractured = glob.glob(os.path.join(DATA_DIR, "Fractured", "*.jpg"))
non_fractured = glob.glob(os.path.join(DATA_DIR, "Non_fractured", "*.jpg"))
all_paths = fractured + non_fractured
all_labels = [1] * len(fractured) + [0] * len(non_fractured)

# Shuffle deterministically
combined = list(zip(all_paths, all_labels))
random.seed(42)
random.shuffle(combined)
all_paths[:], all_labels[:] = zip(*combined)

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# Model definition
def get_model():
    model = timm.create_model("deit_tiny_patch16_224", pretrained=True, num_classes=1)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.get_classifier().parameters():
        param.requires_grad = True

    return model

# Save path for model and plots
if not os.path.exists(PLOT_SAVE_PATH):
    os.makedirs(PLOT_SAVE_PATH)

if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    best_model = None
    best_fold_loss = float('inf')  # Track the best fold loss

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
        print(f"\nFold {fold+1}/{NUM_FOLDS}")

        # Prepare datasets and dataloaders
        train_ds = FractureDataset([all_paths[i] for i in train_idx], [all_labels[i] for i in train_idx], transform)
        val_ds = FractureDataset([all_paths[i] for i in val_idx], [all_labels[i] for i in val_idx], transform)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = get_model().to(DEVICE)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=LEARNING_RATE * 0.1)

        train_losses = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0
            correct = 0
            total = 0

            for inputs, labels in tqdm(train_loader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1).float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            train_losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            if epoch >= SWA_START:
                swa_model.update_parameters(model)
                swa_scheduler.step()

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered")
                    break

        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)

        swa_model.eval()
        all_preds, all_labels_eval, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1).float()
                outputs = swa_model(inputs)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels_eval.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        auc = roc_auc_score(all_labels_eval, all_probs)
        f1 = f1_score(all_labels_eval, all_preds)
        cm = confusion_matrix(all_labels_eval, all_preds)
        acc = (cm[0,0] + cm[1,1]) / cm.sum()

        print(f"Fold {fold+1} Accuracy: {acc*100:.2f}%")
        print(f"AUC: {auc:.4f}, F1 Score: {f1:.4f}")
        print("Confusion Matrix:\n", cm)

        # Save the best model after each fold if it has the lowest loss
        if best_loss < best_fold_loss:
            best_fold_loss = best_loss
            best_model = swa_model.state_dict()

        # Save the loss plot
        plt.figure()
        plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
        plt.title(f'Training Loss Curve - Fold {fold+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_SAVE_PATH, f'train_loss_fold_{fold+1}.png'))
        plt.close()

        # Save the confusion matrix plot
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Fold {fold+1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(PLOT_SAVE_PATH, f'confusion_matrix_fold_{fold+1}.png'))
        plt.close()

    # After all folds are done, save the best model
    if best_model:
        torch.save(best_model, MODEL_SAVE_PATH)
        print(f"Best model saved to {MODEL_SAVE_PATH}")