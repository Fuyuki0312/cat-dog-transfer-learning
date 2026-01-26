from model import DogVsCatWithResNet18
import torch
from PIL import Image
from torchvision import transforms

# Set up before inference ----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DogVsCatWithResNet18().to(device)
transform = transforms.Compose([
    transforms.Resize(265),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# Load model ------------------------------------------

checkpoint = torch.load("DogCatModel.pth", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])

# Inference mode ---------------------------------------

model.eval()
while True:
    user_input = str(input("Enter an image's name: "))
    if user_input == "quit":
        break
    input_image = transform(Image.open(user_input)).to(device).unsqueeze(dim=0)

    with torch.inference_mode():
        model_logits = model(input_image)
    pred_prob = torch.softmax(model_logits, dim=1)
    result = torch.argmax(pred_prob, dim=1).item()
    print("cat" if result == 0 else "dog")