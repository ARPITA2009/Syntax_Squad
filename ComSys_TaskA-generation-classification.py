import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
import time
import copy
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Define paths for input and output directories
DATA_DIR = '/kaggle/input/latest-dataset/Task_A'
MASTER_DIR = '/kaggle/working/Task_A/master'
NEW_TRAIN_DIR = '/kaggle/working/Task_A/train'
NEW_VAL_DIR = '/kaggle/working/Task_A/val'

# Step 1: Combine training and validation datasets into a master dataset
def create_master_dataset(train_dir, val_dir, master_dir):
    """
    Combines images from training and validation directories into a single master dataset.
    Adds '_val' suffix to validation images to avoid filename conflicts.
    Returns True if successful, False otherwise.
    """
    os.makedirs(master_dir, exist_ok=True)

    def copy_images(source_dir, dest_dir, suffix=''):
        if not os.path.exists(source_dir):
            print(f"Error: Source directory {source_dir} does not exist")
            return False
        for gender in ['female', 'male']:
            source_gender_path = os.path.join(source_dir, gender)
            dest_gender_path = os.path.join(dest_dir, gender)
            if not os.path.isdir(source_gender_path):
                print(f"Warning: {source_gender_path} does not exist")
                continue
            os.makedirs(dest_gender_path, exist_ok=True)
            for img_name in os.listdir(source_gender_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    dest_img_name = img_name if not suffix else f"{os.path.splitext(img_name)[0]}{suffix}{os.path.splitext(img_name)[1]}"
                    try:
                        shutil.copy(os.path.join(source_gender_path, img_name), os.path.join(dest_gender_path, dest_img_name))
                    except (shutil.Error, OSError) as e:
                        print(f"Error copying {img_name}: {e}")
        return True

    print("Copying training images to master dataset...")
    if not copy_images(os.path.join(train_dir, 'train'), master_dir):
        return False
    print("Copying validation images to master dataset...")
    if not copy_images(os.path.join(train_dir, 'val'), master_dir, suffix='_val'):
        return False

    # Verify master dataset
    for gender in ['female', 'male']:
        gender_path = os.path.join(master_dir, gender)
        if os.path.isdir(gender_path):
            image_count = len([img for img in os.listdir(gender_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"{gender.capitalize()} images in master dataset: {image_count}")
        else:
            print(f"Error: {gender} folder not found in master dataset")
            return False
    return True

# Create master dataset
print("Creating master dataset...")
if not create_master_dataset(DATA_DIR, DATA_DIR, MASTER_DIR):
    print("Failed to create master dataset. Exiting.")
    exit(1)

# Step 2: Split master dataset into new training (70%) and validation (30%) sets
def split_dataset(master_dir, train_dir, val_dir, train_ratio=0.7):
    """
    Splits the master dataset into training and validation sets based on the train_ratio.
    Returns True if successful, False otherwise.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for gender in ['female', 'male']:
        master_gender_path = os.path.join(master_dir, gender)
        train_gender_path = os.path.join(train_dir, gender)
        val_gender_path = os.path.join(val_dir, gender)
        os.makedirs(train_gender_path, exist_ok=True)
        os.makedirs(val_gender_path, exist_ok=True)

        images = [img for img in os.listdir(master_gender_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"Warning: No images found in {master_gender_path}")
            continue

        random.shuffle(images)
        train_count = int(len(images) * train_ratio)
        train_images = images[:train_count]
        val_images = images[train_count:]

        for img_name in train_images:
            shutil.copy(os.path.join(master_gender_path, img_name), os.path.join(train_gender_path, img_name))
        for img_name in val_images:
            shutil.copy(os.path.join(master_gender_path, img_name), os.path.join(val_gender_path, img_name))

        print(f"Gender: {gender.capitalize()}")
        print(f"  Training images: {len(train_images)}")
        print(f"  Validation images: {len(val_images)}")
    return True

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Split dataset
print("\nSplitting master dataset into train and validation...")
if not split_dataset(MASTER_DIR, NEW_TRAIN_DIR, NEW_VAL_DIR):
    print("Failed to split dataset. Exiting.")
    exit(1)

# Verify split
for split_dir, name in [(NEW_TRAIN_DIR, 'Training'), (NEW_VAL_DIR, 'Validation')]:
    total_images = 0
    for gender in ['female', 'male']:
        gender_path = os.path.join(split_dir, gender)
        if os.path.isdir(gender_path):
            image_count = len([img for img in os.listdir(gender_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
            total_images += image_count
            print(f"{name} - {gender.capitalize()}: {image_count} images")
    print(f"Total {name} images: {total_images}")

# Step 3: Define data transformations with enhanced augmentations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
try:
    image_datasets = {
        'train': datasets.ImageFolder(NEW_TRAIN_DIR, data_transforms['train']),
        'val': datasets.ImageFolder(NEW_VAL_DIR, data_transforms['val']),
    }
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# Print class-to-index mapping
print(f"\nClass to index mapping: {image_datasets['train'].class_to_idx}")

# Create data loaders with weighted sampling for training
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"Classes: {class_names}")
print(f"Training samples: {dataset_sizes['train']}")
print(f"Validation samples: {dataset_sizes['val']}")

# WeightedRandomSampler for balanced sampling
targets = image_datasets['train'].targets
class_counts = np.bincount(targets)  # female: 159, male: 824
class_weights = 1. / class_counts
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, sampler=sampler, num_workers=2 if torch.cuda.is_available() else 0),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=2 if torch.cuda.is_available() else 0),
}

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and modify ResNet50, unfreeze layer4
model = models.resnet50(weights='DEFAULT')
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
model = model.to(device)

# Define loss function with class weights, optimizer, and scheduler
class_weights = torch.tensor([1.0, 159/824]).to(device)  # Weight male class less due to higher count
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Training function with per-class metrics
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    Trains the model with early stopping based on validation performance.
    Computes Accuracy, Precision, Recall, and F1-Score (per-class and weighted) for validation phase.
    Returns the model with the best validation accuracy.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    patience = 7  # Increased patience for early stopping
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'val':
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'val':
                epoch_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
                epoch_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
                epoch_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                print(f'  Precision (Female, Male): {epoch_precision[0]:.4f}, {epoch_precision[1]:.4f}')
                print(f'  Recall (Female, Male): {epoch_recall[0]:.4f}, {epoch_recall[1]:.4f}')
                print(f'  F1-Score (Female, Male): {epoch_f1[0]:.4f}, {epoch_f1[1]:.4f}')
                print(f'  Weighted Avg - Precision: {precision_score(all_labels, all_preds, average="weighted", zero_division=0):.4f}, '
                      f'Recall: {recall_score(all_labels, all_preds, average="weighted", zero_division=0):.4f}, '
                      f'F1: {f1_score(all_labels, all_preds, average="weighted", zero_division=0):.4f}')
                
                if epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss < best_loss):
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping after {epoch + 1} epochs')
                    model.load_state_dict(best_model_wts)
                    print(f'Training completed in {time.time() - start_time:.0f}s')
                    print(f'Best validation Acc: {best_acc:.4f}, Best validation Loss: {best_loss:.4f}')
                    return model
                scheduler.step(epoch_loss)
            else:
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print(f'Training completed in {time.time() - start_time:.0f}s')
    print(f'Best validation Acc: {best_acc:.4f}, Best validation Loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# Train model
print("\nStarting training with ResNet50...")
model = train_model(model, criterion, optimizer, scheduler)

# Save model
try:
    torch.save(model.state_dict(), '/kaggle/working/resnet50_task_a.pth')
    print("Model saved to /kaggle/working/resnet50_task_a.pth")
except Exception as e:
    print(f"Error saving model: {e}")
