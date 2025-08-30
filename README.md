# 🐶🐱 Cat vs Dog Classifier

A simple image classification app built with **FastAI**, **Gradio**, and a model hosted on **Hugging Face Hub**.  
This app classifies images as either a **Dog** or a **Cat**.

---

## Table of Contents

1. [Overview](#overview)  
2. [Model Details](#model-details)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Project Structure](#project-structure)  
6. [Notes](#notes)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## Overview

This project demonstrates a simple **computer vision pipeline**:

1. A **FastAI learner** model is trained to classify images into "Cat" or "Dog".
2. The model is **hosted on Hugging Face Hub**.
3. A **Gradio web interface** is used to allow users to upload images and get predictions.

The app downloads the model dynamically from Hugging Face Hub, so you **do not need to include the large model file in GitHub**.

---

## Model Details

- **Model type:** FastAI `Learner`  
- **Training dataset:** Cat and Dog images (your training data)  
- **Hosted on:** [Hugging Face Hub](https://huggingface.co/Kutti-AI/cat-dog)  
- **Loading code:**

```python(version must be 2.10 or 2.18 or 2 series)
from fastai.vision.all import load_learner
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Kutti-AI/cat-dog",
    filename="model_fastai.pkl"
)
learn = load_learner(model_path)
