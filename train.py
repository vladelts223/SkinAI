import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

CSV_FILE = 'dataset/fitzpatrick17k.csv'
IMG_DIR = 'dataset/fitzpatrick_images'
MODEL_NAME = 'vit_small_patch16_224'
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FitzpatrickDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.data = df[df['fitzpatrick_scale'].between(1, 6)].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_hash = self.data.iloc[idx]['md5hash']
        label = int(self.data.iloc[idx]['fitzpatrick_scale']) - 1

        img_path = os.path.join(self.img_dir, str(label + 1), f"{img_hash}.jpg")

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = FitzpatrickDataset(CSV_FILE, IMG_DIR, transform=transform)
train_size = int(0.85 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=6)
model.to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

scaler = torch.amp.GradScaler('cuda')

history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}


def train():
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        t_loss, t_correct = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            t_correct += (pred == lbls).sum().item()
            pbar.set_postfix({'acc': t_correct / ((pbar.n + 1) * BATCH_SIZE)})

        model.eval()
        v_loss, v_correct = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    out = model(imgs)
                    loss = criterion(out, lbls)
                v_loss += loss.item()
                _, pred = torch.max(out, 1)
                v_correct += (pred == lbls).sum().item()

        train_acc = t_correct / len(train_ds)
        val_acc = v_correct / len(val_ds)

        print(f"Epoch {epoch + 1} Results: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(t_loss / len(train_loader))
        history['val_loss'].append(v_loss / len(val_loader))

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'fitzpatrick_vit_small_best.pth')
            print(">>> Модель оновлено та збережено!")


def plot_history():
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('fitzpatrick_results.png')
    print("Графіки збережено у 'fitzpatrick_results.png'")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nНавчання зупинено користувачем.")
    finally:
        plot_history()