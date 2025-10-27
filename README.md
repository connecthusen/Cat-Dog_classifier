# 🐶🐱 Cat vs Dog Classifier

A simple yet powerful **image classification model** built using **fast.ai** and deployed with **Gradio** + **Hugging Face Spaces**.  
This model predicts whether an uploaded image is of a **Cat 🐱** or a **Dog 🐶**.

---

## 🧠 Model Details

- **Framework:** fast.ai (PyTorch backend)  
- **Base Model:** `resnet34` (transfer learning)  
- **Dataset:** Custom Cat vs Dog dataset  
- **Trained on:** Google Colab / Jupyter Notebook  
- **Output Classes:** `"Cat"` and `"Dog"`

---

## 🚀 Demo

👉 **Try it Live:** [Cat vs Dog Classifier on Hugging Face Spaces](https://huggingface.co/spaces/Kutti-AI/cat-dog)

---

## 🗂️ Repository Structure
```text

📦 cat-dog/
┣ 📜 app.py # Gradio app (main script)
┣ 📜 cat.jpg # image file
┣ 📜 cat2.jpg # image file
┣ 📜 dog.jpg # image file
┣ 📜 dog2.jpg # image file
┣ 📜 requirements.txt # Dependencies for Hugging Face Space
┗ 📜 README.md # Project documentation
```

---



## 🧩 How It Works

1. The model is automatically downloaded from Hugging Face Hub using:
   ```python
   from huggingface_hub import hf_hub_download
   model_path = hf_hub_download(repo_id="Kutti-AI/cat-dog", filename="model_fastai.pkl")
The model is loaded with fastai.load_learner().

A simple Gradio interface allows uploading an image to get predictions instantly.

💻 Run Locally
To run this app on your own system:

# Clone the repository
git clone https://huggingface.co/spaces/Kutti-AI/cat-dog
cd cat-dog

# Install dependencies
pip install -r requirements.txt

# Launch the app
python app.py

Then open your browser and visit:

http://127.0.0.1:7860

---

## 🧰 Requirements

fastai==2.7.13
fastcore==1.5.55
torch==2.1.2
torchvision==0.16.2
transformers==4.40.2
datasets==2.13.1
numpy==1.24.4
pandas==2.2.3
matplotlib==3.7.2
spacy==3.8.7
gradio==4.44.1

---

## 🧪 Example Predictions

Image	Prediction

🐶 dog.jpg	Dog → 0.98
🐱 cat.jpg	Cat → 0.97

---

## 🧠 Training Details

Below is a simplified version of how this model was trained using fast.ai:

python

from fastai.vision.all import *

# 1️⃣ Define dataset path
path = Path('data/cat_dog')

# 2️⃣ Create DataLoaders
dls = ImageDataLoaders.from_folder(
    path,
    train='train',
    valid='valid',
    item_tfms=Resize(224),
    batch_tfms=aug_transforms()
)

# 3️⃣ Define and train model
learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(3)

# 4️⃣ Export trained model
learn.export('model_fastai.pkl')
✅ Transfer Learning:
Used resnet34 pretrained on ImageNet.
✅ Augmentations:
Used fast.ai’s built-in transforms (flip, rotate, zoom).
✅ Validation Accuracy: ~98%

---

## 🧠 Inference Code Example

from fastai.vision.all import load_learner
from huggingface_hub import hf_hub_download
import gradio as gr

# Download model from Hugging Face Hub
model_path = hf_hub_download(repo_id="Kutti-AI/cat-dog", filename="model_fastai.pkl")
learn = load_learner(model_path)

categories = ("Dog", "Cat")

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(type="pil")
label = gr.Label(num_top_classes=2)

intf = gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    title="🐶🐱 Cat vs Dog Classifier",
    description="Upload an image of a cat or dog and get predictions."
)

if __name__ == "__main__":
    intf.launch()

---

## 📜 License

This project is released under the MIT License — feel free to use, remix, and share it with attribution.

---

## ❤️ Author

Created by Husen (Kutti-AI)
Made with 💕 using fast.ai, Gradio, and Hugging Face ✨

---
