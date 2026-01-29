from model import DogVsCatWithResNet18

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DogVsCatWithResNet18().to(device)

# Hyperparameters -------------------------------------------------------

TRAINING_CYCLES = 4
BATCH_SIZE = 64
TEST_AFTER_n_TRAINING_CYCLES = 1
torch.manual_seed(42)

# Loss, accuracy, scheduler before training -----------------------------

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.fc.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                       factor=0.5,
                                                       patience=2,
                                                       threshold=0.005,
                                                       min_lr=0.0000001)

def accuracy_func(pred, true):
    acc_tensor = torch.eq(true, torch.argmax(pred, dim=1))
    acc = torch.sum(acc_tensor) / len(true)
    return acc * 100

# Load train data -------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(265),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))  # For cat dog classification
])

full_data = datasets.ImageFolder("CatVsDog",
                                  transform=transform)

train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size

train_data, test_data = random_split(
    full_data,
    [train_size, test_size]
)

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    shuffle=True,
    num_workers=4
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    shuffle=False,
    num_workers=4
)

def main():

    # Load model -------------------------------------------------------

    try:
        checkpoint = torch.load("DogCatModel.pth", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
    except Exception as e:
        print(e)
        print("Trying to create a new model")
        epoch = 0

    # Training loop ------------------------------------------------------

    epochs = epoch + TRAINING_CYCLES

    for epoch in range(epoch+1, epochs+1):
        if epoch % TEST_AFTER_n_TRAINING_CYCLES == 0:
            print("Processing Epoch " + str(epoch) + "...")
        sum_loss, sum_acc = 0, 0
      
        model.train()
        for images, labels in train_dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            pred_logits = model(images)

            batch_loss = loss_func(pred_logits, labels)
            batch_acc = accuracy_func(pred=pred_logits, true=labels)

            optimizer.zero_grad()

            batch_loss.backward()

            optimizer.step()

            sum_loss += batch_loss
            sum_acc += batch_acc

        loss = (sum_loss) / len(train_dataloader)
        acc = (sum_acc) / len(train_dataloader)
        scheduler.step(loss)

        if epoch % TEST_AFTER_n_TRAINING_CYCLES == 0:
            test_sum_loss, test_sum_acc = 0, 0

            model.eval()
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)

                with torch.inference_mode():
                    test_pred_logits = model(images)
                test_batch_loss = loss_func(test_pred_logits, labels)
                test_batch_acc = accuracy_func(test_pred_logits, labels)

                test_sum_loss += test_batch_loss
                test_sum_acc += test_batch_acc

            test_loss = test_sum_loss / len(test_dataloader)
            test_acc = test_sum_acc / len(test_dataloader)
                
            print(f"Epoch: {epoch} | Loss: {loss:.6f} | Accuracy: {acc:.1f}%")
            print(f"         Test loss: {test_loss:.6f} | Test accuracy: {test_acc:.1f}%")

    # Save model -------------------------------------------------------

    torch.save(obj={"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch},
               f="DogCatModel.pth")

if __name__ == '__main__':
    main()
