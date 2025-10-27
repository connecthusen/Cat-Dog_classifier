from fastai.vision.all import load_learner
from huggingface_hub import hf_hub_download
import gradio as gr

# ===============================
# 1️⃣ Download model from HF Hub
# ===============================
# Change repo_id to your own repo on Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="Kutti-AI/cat-dog",  # <--- replace with your repo
    filename="model_fastai.pkl"
)

def is_cat(x): return x[0].isupper()
    
# Load learner
learn = load_learner(model_path)

# ===============================
# 2️⃣ Define prediction function
# ===============================
categories = ("Dog", "Cat")

def classify_image(im):
    learn.model.eval()  # Force evaluation mode
    if not isinstance(im, Image.Image):
        im = Image.fromarray(im)
    with learn.no_bar():
        pred, idx, probs = learn.predict(im)
    return dict(zip(categories, map(float, probs)))

# ===============================
# 3️⃣ Gradio UI
# ===============================
image = gr.Image(type="pil")
label = gr.Label(num_top_classes=2)
examples = ["dog.jpg", "cat.jpg", "dog2.jpg", "cat2.jpg"]

intf = gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    examples=examples,
    title="🐶🐱 Cat vs Dog Classifier",
    description="Upload an image of a cat or dog and get predictions."
)

if __name__ == "__main__":
    intf.launch()
