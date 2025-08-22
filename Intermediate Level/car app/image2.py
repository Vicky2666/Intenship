import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gradio as gr
import urllib.request

# Load pretrained MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = urllib.request.urlopen(LABELS_URL)
labels = [line.decode("utf-8").strip() for line in response.readlines()]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Prediction function with error handling
def classify_image(img):
    try:
        if img is None:
            return "No image provided!"
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = outputs.max(1)
        return labels[predicted.item()]
    except Exception as e:
        return f"Error: {e}"

# Gradio Blocks interface
with gr.Blocks(css=".gradio-container {background-color: #f0f8ff}") as demo:
    gr.Markdown("## 🌟 MobileNetV2 Image Classifier")
    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Image")
        output_text = gr.Textbox(label="Prediction", interactive=False)
    btn = gr.Button("Classify")
    btn.click(fn=classify_image, inputs=img_input, outputs=output_text)

# Launch interface
demo.launch(share=True)