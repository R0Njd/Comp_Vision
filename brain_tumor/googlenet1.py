import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  
from torchvision.models import googlenet, GoogLeNet_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(r'C:\Users\ronjd\OneDrive\Desktop\comp_V\brain_tumor\Training', transform=data_transforms)
test_dataset = ImageFolder(r'C:\Users\ronjd\OneDrive\Desktop\comp_V\brain_tumor\Testing', transform=data_transforms)
size = len(train_dataset)
train_size = int(0.85 * size) 
val_size = size - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#Visualization
IMAGE_PER_CATEGORY = 4
CATEGORIES = train_dataset.classes
NUM_OF_CATEGORIES = len(CATEGORIES)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# Prepare image samples per category
category_images = {cat: [] for cat in CATEGORIES}

for images, labels in train_loader:
    for img, label in zip(images, labels):
        cat = CATEGORIES[label]
        if len(category_images[cat]) < IMAGE_PER_CATEGORY:
            category_images[cat].append(img)
    if all(len(imgs) >= IMAGE_PER_CATEGORY for imgs in category_images.values()):
        break

fig = plt.figure(figsize=(12, 8))
grid = ImageGrid(fig, 111, nrows_ncols=(IMAGE_PER_CATEGORY, NUM_OF_CATEGORIES), axes_pad=0.2)

i = 0
for col, category in enumerate(CATEGORIES):
    for row in range(IMAGE_PER_CATEGORY):
        ax = grid[i]
        img_tensor = category_images[category][row]
        img = img_tensor.numpy().transpose((1, 2, 0))
        img = (img * std + mean).clip(0, 1)
        ax.imshow(img)
        ax.axis('off')
        if row == IMAGE_PER_CATEGORY - 1:
            ax.set_title(category, fontsize=10, backgroundcolor='white')
        i += 1

plt.show()

num_classes = len(train_dataset.classes)

model = models.googlenet(pretrained=True)

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 🔽 Calculate batch accuracy
        batch_acc = 100 * predicted.eq(labels).sum().item() / labels.size(0)

        # 🔽 Update progress bar with loss and batch accuracy
        progress_bar.set_postfix(loss=loss.item(), batch_acc=f"{batch_acc:.2f}%")

    # 🔽 Final epoch-level training accuracy
    train_acc = 100 * correct / total

    # Validation after epoch
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            _, val_preds = val_outputs.max(1)
            val_total += val_labels.size(0)
            val_correct += val_preds.eq(val_labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
 
# Final Test Accuracy
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for test_inputs, test_labels in test_loader:
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)
        test_outputs = model(test_inputs)
        _, test_preds = test_outputs.max(1)
        test_total += test_labels.size(0)
        test_correct += test_preds.eq(test_labels).sum().item()

test_acc = 100 * test_correct / test_total
print(f"Final Test Accuracy:        {test_acc:.2f}%")


# Train and test
train_model(model, train_loader, val_loader, epochs=5)
test_acc = evaluate(model, test_loader)
print(f"\n\U0001F9EA Final Test Accuracy: {test_acc:.2f}%")