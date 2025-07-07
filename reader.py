# standard Python
import os
import sys

# NumPy
import numpy as np

# ML
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from dataset import MNISTDigitDataset
from model import DigitCNN
from parser import parseIdx2Np

# metrics and plotting
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# define paths
MODEL_PATH = "digit_cnn.pth"
train_images_path = 'data/train-images-idx3-ubyte'
test_images_path = 'data/t10k-images-idx3-ubyte'
train_labels_path = 'data/train-labels-idx1-ubyte'
test_labels_path = 'data/t10k-labels-idx1-ubyte'

# set device up and connect to model
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model = DigitCNN().to(device)


# call function
training_data, testing_data, training_labels, testing_labels = parseIdx2Np(train_images_path, test_images_path, train_labels_path, test_labels_path)

# check for load failure
if training_data is None:
    print("\nAborting program: Cannot proceed without data files.")
    sys.exit(1) # exit the script with error code 1

# calculate, to feed to transforms, and z-normalize
# we use training data statistics because the model
# learns in the paradigm of the training data, and
# will then act on unseen data based on experience
# we maintain the same transformation always
# convert to float32 [0..1]
training_data_float = training_data.astype(np.float32) / 255.0

# Calculate mean and std on scaled data (float [0..1])
mean_val_train = training_data_float.mean()
std_val_train = training_data_float.std()

print("Mean (scaled):", mean_val_train)
print("Std (scaled):", std_val_train)

# create transform object
transform = v2.Compose([
    v2.ToImage(), # turn from NumPy array into tv_tensors.Image
    v2.ToDtype(torch.float32, scale=True), # convert data type and normalize to [0, 1]
    v2.Normalize(mean=[mean_val_train], std=[std_val_train]) # normalize for mean=0 st.dev.=1 on the only channel,
    # it goes channel by channel, so use list
])

# load datasets PyTorch
train_dataset = MNISTDigitDataset(training_data, training_labels, transform=transform)
test_dataset = MNISTDigitDataset(testing_data, testing_labels, transform=transform)

# define batch size
BATCH_SIZE = 128
    
# shuffle the data to avoid learning about artificial patterns, and be
# stuck on sequential overfitting
train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# skip training when have saved model
if not os.path.exists(MODEL_PATH):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # one epoch -> fully utilised training data once
    EPOCHS = 16

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_data_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # reset gradient
            optimizer.zero_grad()

            # extract logits
            outputs = model(images)

            # apply loss criterion
            loss = criterion(outputs, labels)

            # perform backpropagation
            # use gradient values with optimizer
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # turn to evaluation mode
    model.eval()

    # save the model after training
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# store predicted and true from test
all_preds = []
all_labels = []

# no gradient, just inference
with torch.no_grad():
    for batch in test_data_loader:
        # get images and labels from batch and move to GPU
        images, labels = batch
        images = images.to(device)

        # run through CNN and get raw scores
        outputs = model(images)
        
        # get the predicted digit and move to cpu
        preds = torch.argmax(outputs, dim=1).cpu()

        # convert batch predictions and labels to NumPy
        # arrays and append to total predictions
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# overall metrics
accuracy = accuracy_score(all_labels, all_preds)
print(f"\nOverall Accuracy: {accuracy:.4f}")

# per-class metrics
report = classification_report(all_labels, all_preds, digits=4)
print("\nClassification Report (Per Class):")
print(report)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
