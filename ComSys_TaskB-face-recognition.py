import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import time
import copy
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score

# Attempt to install facenet-pytorch
try:
    import facenet_pytorch
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
    print("facenet-pytorch is available")
except ImportError:
    print("Installing facenet-pytorch...")
    try:
        os.system("pip install facenet-pytorch")
        from facenet_pytorch import InceptionResnetV1
        FACENET_AVAILABLE = True
        print("facenet-pytorch installed successfully")
    except Exception as e:
        print(f"Failed to install facenet-pytorch: {e}")
        print("Falling back to ResNet50")
        FACENET_AVAILABLE = False

# Define paths
DATA_DIR = '/kaggle/input/latest-dataset/Task_A'
MASTER_DIR = '/kaggle/working/Task_A/master'
NEW_TRAIN_DIR = '/kaggle/working/Task_A/train'
NEW_VAL_DIR = '/kaggle/working/Task_A/val'

# Step 1: Combine training and validation datasets into a master dataset
def create_master_dataset(train_dir, val_dir, master_dir):
    os.makedirs(master_dir, exist_ok=True)
    def copy_images(source_dir, dest_dir, suffix=''):
        if not os.path.exists(source_dir):
            print(f"Error: Source directory {source_dir} does not exist")
            return False
        for identity in os.listdir(source_dir):
            source_identity_path = os.path.join(source_dir, identity)
            dest_identity_path = os.path.join(dest_dir, identity)
            if not os.path.isdir(source_identity_path):
                continue
            os.makedirs(dest_identity_path, exist_ok=True)
            for img_name in os.listdir(source_identity_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    dest_img_name = img_name if not suffix else f"{os.path.splitext(img_name)[0]}{suffix}{os.path.splitext(img_name)[1]}"
                    try:
                        shutil.copy(os.path.join(source_identity_path, img_name), os.path.join(dest_identity_path, dest_img_name))
                    except (shutil.Error, OSError) as e:
                        print(f"Error copying {img_name}: {e}")
        return True
    print("Copying training images to master dataset...")
    if not copy_images(os.path.join(train_dir, 'train'), master_dir):
        return False
    print("Copying validation images to master dataset...")
    if not copy_images(os.path.join(train_dir, 'val'), master_dir, suffix='_val'):
        return False
    for identity in os.listdir(master_dir):
        identity_path = os.path.join(master_dir, identity)
        if os.path.isdir(identity_path):
            image_count = len([img for img in os.listdir(identity_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"Identity {identity} images in master dataset: {image_count}")
        else:
            print(f"Error: {identity} folder not found in master dataset")
            return False
    return True

# Create master dataset
print("Creating master dataset...")
if not create_master_dataset(DATA_DIR, DATA_DIR, MASTER_DIR):
    print("Failed to create master dataset. Exiting.")
    exit(1)

# Step 2: Split master dataset into training (70%) and validation (30%) sets
def split_dataset(master_dir, train_dir, val_dir, train_ratio=0.7):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    for identity in os.listdir(master_dir):
        master_identity_path = os.path.join(master_dir, identity)
        train_identity_path = os.path.join(train_dir, identity)
        val_identity_path = os.path.join(val_dir, identity)
        os.makedirs(train_identity_path, exist_ok=True)
        os.makedirs(val_identity_path, exist_ok=True)
        images = [img for img in os.listdir(master_identity_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"Warning: No images found in {master_identity_path}")
            continue
        random.shuffle(images)
        train_count = int(len(images) * train_ratio)
        train_images = images[:train_count]
        val_images = images[train_count:]
        for img_name in train_images:
            shutil.copy(os.path.join(master_identity_path, img_name), os.path.join(train_identity_path, img_name))
        for img_name in val_images:
            shutil.copy(os.path.join(master_identity_path, img_name), os.path.join(val_identity_path, img_name))
        print(f"Identity: {identity}")
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

# Step 3: Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Custom Dataset for Triplets with Hard Mining
class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, model=None, device='cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.model = model
        self.device = device
        self.identities = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.image_paths = []
        self.identity_to_images = {}
        self.embedding_cache = None
        for identity in self.identities:
            identity_path = os.path.join(root_dir, identity)
            images = [os.path.join(identity_path, img) for img in os.listdir(identity_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) >= 2:
                self.identity_to_images[identity] = images
                self.image_paths.extend([(img, identity) for img in images])
        print(f"Found {len(self.identities)} identities with sufficient images")

    def update_embedding_cache(self):
        if self.model is None:
            return
        self.model.eval()
        self.embedding_cache = {}
        with torch.no_grad():
            for img_path, identity in self.image_paths:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img).unsqueeze(0).to(self.device)
                emb = self.model(img).cpu().numpy()
                self.embedding_cache[img_path] = emb
        print("Updated embedding cache for hard triplet mining")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_path, anchor_id = self.image_paths[idx]
        positive_path = random.choice([p for p in self.identity_to_images[anchor_id] if p != anchor_path])
        negative_id = random.choice([i for i in self.identities if i != anchor_id])
        
        if self.embedding_cache and random.random() < 0.5:
            anchor_emb = self.embedding_cache[anchor_path]
            min_dist = float('inf')
            hard_negative_path = None
            for neg_id in self.identities:
                if neg_id == anchor_id:
                    continue
                for neg_path in self.identity_to_images[neg_id]:
                    neg_emb = self.embedding_cache[neg_path]
                    dist = 1 - cosine_similarity(anchor_emb, neg_emb)[0][0]
                    if dist < min_dist:
                        min_dist = dist
                        hard_negative_path = neg_path
            negative_path = hard_negative_path if hard_negative_path else random.choice(self.identity_to_images[negative_id])
        else:
            negative_path = random.choice(self.identity_to_images[negative_id])
        
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img

# Load datasets
try:
    image_datasets = {
        'train': TripletFaceDataset(NEW_TRAIN_DIR, data_transforms['train']),
        'val': TripletFaceDataset(NEW_VAL_DIR, data_transforms['val']),
    }
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# Create data loaders
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=2 if torch.cuda.is_available() else 0),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=2 if torch.cuda.is_available() else 0),
}

print(f"Training samples: {dataset_sizes['train']}")
print(f"Validation samples: {dataset_sizes['val']}")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Embedding Model
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        if FACENET_AVAILABLE:
            self.backbone = InceptionResnetV1(pretrained='vggface2')
            self.input_dim = 512
        else:
            self.backbone = models.resnet50(weights='DEFAULT')
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.input_dim = 2048
        self.bn = nn.BatchNorm1d(self.input_dim)
        self.fc = nn.Linear(self.input_dim, 256)
        self.normalize = lambda x: x / torch.norm(x, dim=1, keepdim=True)

    def forward(self, x):
        x = self.backbone(x)
        if not FACENET_AVAILABLE:
            x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.fc(x)
        x = self.normalize(x)
        return x

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# Training function
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, image_datasets, num_epochs=50):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 10
    epochs_no_improve = 0
    start_time = time.time()

    # Initialize dataset with model for hard triplet mining
    image_datasets['train'].model = model
    image_datasets['train'].device = device

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        if epoch % 5 == 0 and epoch > 0:
            image_datasets['train'].update_embedding_cache()
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            num_triplets = 0
            all_preds = []
            all_labels = []

            for anchors, positives, negatives in dataloaders[phase]:
                anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    anchor_emb = model(anchors)
                    positive_emb = model(positives)
                    negative_emb = model(negatives)
                    loss = criterion(anchor_emb, positive_emb, negative_emb)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if phase == 'val':
                    pos_sim = torch.cosine_similarity(anchor_emb, positive_emb)
                    neg_sim = torch.cosine_similarity(anchor_emb, negative_emb)
                    preds_tensor = (pos_sim > neg_sim).long()
                    labels_tensor = torch.ones_like(preds_tensor)
                    preds = preds_tensor.cpu().numpy()
                    labels = labels_tensor.cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)

                running_loss += loss.item() * anchors.size(0)
                num_triplets += anchors.size(0)

            epoch_loss = running_loss / num_triplets
            if phase == 'val':
                epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
                epoch_f1 = f1_score(all_labels, all_preds, average='macro')
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, F1-Score: {epoch_f1:.4f}')
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping after {epoch + 1} epochs')
                    model.load_state_dict(best_model_wts)
                    print(f'Training completed in {time.time() - start_time:.0f}s')
                    print(f'Best validation Accuracy: {best_acc:.4f}')
                    return model
            else:
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}')
        print('-' * 10)

    print(f'Training completed in {time.time() - start_time:.0f}s')
    print(f'Best validation Accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# Inference function
def evaluate_model(model, val_dir):
    model.eval()
    reference_embeddings = {}
    for identity in os.listdir(val_dir):
        identity_path = os.path.join(val_dir, identity)
        if not os.path.isdir(identity_path):
            continue
        embeddings = []
        for img_name in os.listdir(identity_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(identity_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img = data_transforms['val'](img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(img).cpu().numpy()
                embeddings.append(emb)
        reference_embeddings[identity] = np.mean(embeddings, axis=0)
    
    correct = 0
    total = 0
    true_labels = []
    pred_labels = []
    
    for identity in os.listdir(val_dir):
        identity_path = os.path.join(val_dir, identity)
        if not os.path.isdir(identity_path):
            continue
        for img_name in os.listdir(identity_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(identity_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img = data_transforms['val'](img).unsqueeze(0).to(device)
                with torch.no_grad():
                    test_emb = model(img).cpu().numpy()
                min_dist = float('inf')
                predicted_identity = None
                for ref_id, ref_emb in reference_embeddings.items():
                    dist = 1 - cosine_similarity(test_emb, ref_emb)[0][0]
                    if dist < min_dist:
                        min_dist = dist
                        predicted_identity = ref_id
                true_labels.append(identity)
                pred_labels.append(predicted_identity)
                if predicted_identity == identity:
                    correct += 1
                total += 1
    
    top1_accuracy = correct / total if total > 0 else 0
    macro_f1 = f1_score(true_labels, pred_labels, average='macro') if total > 0 else 0
    
    print(f'Top-1 Validation Accuracy: {top1_accuracy:.4f}')
    print(f'Macro-averaged F1-Score: {macro_f1:.4f}')

# Initialize model, criterion, optimizer, scheduler
model = EmbeddingNet().to(device)
criterion = TripletLoss(margin=1.2)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Train model
print("\nStarting training with Triplet Network...")
model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, image_datasets)

# Save model
try:
    model_save_path = '/kaggle/working/triplet_inceptionresnet.pth' if FACENET_AVAILABLE else '/kaggle/working/triplet_resnet50.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
except Exception as e:
    print(f"Error saving model: {e}")

# Evaluate model
print("\nEvaluating model...")
evaluate_model(model, NEW_VAL_DIR)
