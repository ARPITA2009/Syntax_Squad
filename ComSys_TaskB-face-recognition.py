import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define directories
input_train_dir = '/kaggle/input/latest-dataset/Task_B/train'
input_val_dir = '/kaggle/input/latest-dataset/Task_B/val'
master_dir = '/kaggle/working/Task_B/master'
new_train_dir = '/kaggle/working/Task_B/new_train'
new_val_dir = '/kaggle/working/Task_B/new_val'
os.makedirs(master_dir, exist_ok=True)
os.makedirs(new_train_dir, exist_ok=True)
os.makedirs(new_val_dir, exist_ok=True)

# Function to copy images to master dataset
def copy_images_to_master(source_dir, master_dir, suffix=''):
    for person_folder in os.listdir(source_dir):
        source_person_path = os.path.join(source_dir, person_folder)
        if os.path.isdir(source_person_path):
            master_person_path = os.path.join(master_dir, person_folder)
            os.makedirs(master_person_path, exist_ok=True)
            
            # Copy normal images
            for img_name in os.listdir(source_person_path):
                img_path = os.path.join(source_person_path, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    dest_img_name = img_name if not suffix else f"{os.path.splitext(img_name)[0]}{suffix}{os.path.splitext(img_name)[1]}"
                    shutil.copy(img_path, os.path.join(master_person_path, dest_img_name))
            
            # Copy distorted images
            distortion_path = os.path.join(source_person_path, 'distortion')
            if os.path.exists(distortion_path):
                master_distortion_path = os.path.join(master_person_path, 'distortion')
                os.makedirs(master_distortion_path, exist_ok=True)
                for img_name in os.listdir(distortion_path):
                    img_path = os.path.join(distortion_path, img_name)
                    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        dest_img_name = img_name if not suffix else f"{os.path.splitext(img_name)[0]}{suffix}{os.path.splitext(img_name)[1]}"
                        shutil.copy(img_path, os.path.join(master_distortion_path, dest_img_name))

# Function to split master dataset into train and validation
def split_master_dataset(master_dir, new_train_dir, new_val_dir, train_ratio=0.7):
    for person_folder in os.listdir(master_dir):
        person_path = os.path.join(master_dir, person_folder)
        if os.path.isdir(person_path):
            # Create person folders in new train and val directories
            train_person_path = os.path.join(new_train_dir, person_folder)
            val_person_path = os.path.join(new_val_dir, person_folder)
            os.makedirs(train_person_path, exist_ok=True)
            os.makedirs(val_person_path, exist_ok=True)
            
            # Get normal images
            normal_images = [
                f for f in os.listdir(person_path)
                if os.path.isfile(os.path.join(person_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            # Get distorted images
            distortion_path = os.path.join(person_path, 'distortion')
            distorted_images = [
                f for f in os.listdir(distortion_path)
                if os.path.isfile(os.path.join(distortion_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ] if os.path.exists(distortion_path) else []
            
            # Combine and shuffle all images
            all_images = normal_images + [f"distortion/{f}" for f in distorted_images]
            random.shuffle(all_images)
            
            # Split images
            train_count = int(len(all_images) * train_ratio)
            if len(all_images) == 1:  # Handle single-image cases
                train_count = 1
                train_images = all_images[:train_count]
                val_images = all_images[:train_count]  # Duplicate single image
            else:
                train_images = all_images[:train_count]
                val_images = all_images[train_count:]
            
            # Copy training images
            train_distortion_path = os.path.join(train_person_path, 'distortion')
            os.makedirs(train_distortion_path, exist_ok=True)
            for img in train_images:
                src_path = os.path.join(master_dir, person_folder, img)
                dest_path = os.path.join(new_train_dir, person_folder, img)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(src_path, dest_path)
            
            # Copy validation images
            val_distortion_path = os.path.join(val_person_path, 'distortion')
            os.makedirs(val_distortion_path, exist_ok=True)
            for img in val_images:
                src_path = os.path.join(master_dir, person_folder, img)
                dest_path = os.path.join(new_val_dir, person_folder, img)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(src_path, dest_path)

# Function to train the model with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, patience=5):
    model.train()
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = '/kaggle/working/best_face_recognition_model.pth'
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model with Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

# Function to evaluate the model
def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    top1_accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return top1_accuracy, macro_f1

# Main execution
if __name__ == "__main__":
    # Step 1: Create master dataset
    print("Copying training images to master dataset...")
    copy_images_to_master(input_train_dir, master_dir)
    print("Copying validation images to master dataset...")
    copy_images_to_master(input_val_dir, master_dir, suffix='_val')
    
    # Verify master dataset
    master_folders = [f for f in os.listdir(master_dir) if os.path.isdir(os.path.join(master_dir, f))]
    print(f"Total person folders in master dataset: {len(master_folders)}")
    print("Sample person folders:", master_folders[:5])
    
    # Step 2: Split master dataset
    print("Splitting master dataset into new train and validation sets...")
    split_master_dataset(master_dir, new_train_dir, new_val_dir)
    
    # Verify split
    train_folders = [f for f in os.listdir(new_train_dir) if os.path.isdir(os.path.join(new_train_dir, f))]
    val_folders = [f for f in os.listdir(new_val_dir) if os.path.isdir(os.path.join(new_val_dir, f))]
    common_folders = set(train_folders).intersection(set(val_folders))
    print(f"Training folders: {len(train_folders)}")
    print(f"Validation folders: {len(val_folders)}")
    print(f"Common folders: {len(common_folders)}")
    print("Sample common folders:", sorted(list(common_folders))[:5])
    
    # Step 3: Define image transformations and load datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(new_train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(new_val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Step 4: Initialize and train ResNet-50 model
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training model...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, patience=5)
    
    # Step 5: Load best model and evaluate
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load('/kaggle/working/best_face_recognition_model.pth'))
    print("Evaluating model...")
    top1_accuracy, macro_f1 = evaluate_model(model, val_loader)
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Macro-averaged F1-Score: {macro_f1:.4f}")
    
    # Save the final model
    torch.save(model.state_dict(), '/kaggle/working/face_recognition_model.pth')
