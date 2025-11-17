import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models import SimpleCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.title("Reconnaissance de chiffres MNIST")

uploaded_file = st.file_uploader("Choisir une image PNG/JPG", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Image chargée", use_column_width=True)
    
    
    image_tensor = transform(image).unsqueeze(0).to(device) 

    
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, 1).item()
    
    st.write(f"Le chiffre prédit est : **{pred}**")
