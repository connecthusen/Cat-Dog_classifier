from fastai.vision.all import load_learner
from huggingface_hub import login, hf_hub_download  
import gradio as gr
from PIL import Image
import os


hf_token = os.environ.get("hf_token")

if hf_token is not None:
    login(token=hf_token)
else:
    raise ValueError("HUGGINGFACE_HUB_TOKEN not set in environment variables")

# Download model from the correct repo
model_path = hf_hub_download(
    repo_id="Kutti-AI/cat-dog",  # ✅ correct repo
    filename="model_fastai.pkl"              # ✅ correct file
)

def is_cat(x):return x[0].isupper()

learn=load_learner(model_path)

categories=('Dog','Cat')
def classify_image(im):
    learn.model.eval()  # Force evaluation mode
    if not isinstance(im, Image.Image):
        im = Image.fromarray(im)
    with learn.no_bar():
        pred, idx, probs = learn.predict(im)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(type="pil")
label = gr.Label(num_top_classes=2)
examples=['dog.jpg','cat.jpg','dog2.jpg','cat2.jpg']

intf = gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    examples=examples
)
intf.launch(share=True)
