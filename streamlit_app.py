# ✅ streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# ==============================
# 🔹 Model Definition (same as training)
# ==============================
class MaskCNN(nn.Module):
    def __init__(self):
        super(MaskCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 2 classes
        )

    def forward(self, x):
        return self.model(x)

# ==============================
# 🔹 Load Model
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaskCNN().to(device)
model.load_state_dict(torch.load("mask_cnn.pth", map_location=device))
model.eval()

# ✅ Class labels (set same as training dataset)
classes = ['with_mask', 'without_mask']

# ✅ Image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ==============================
# 🔹 Streamlit UI
# ==============================
st.title("😷 Mask Classifier ")
st.write("Upload an image and let the model predict the class.")

# ✅ File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

    # Show result
    st.subheader(f"Prediction: **{label}** ✅")
