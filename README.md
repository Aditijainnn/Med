# Medical Prescription OCR 

A transformer-based Optical Character Recognition (OCR) system for handwritten medical prescriptions, built on **NAVER Clova Donut** and enhanced with zero-shot document classification.


---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Features](#-features)
3. [Performance](#-performance)
4. [Quick Start](#-quick-start)
5. [Model Usage](#-model-usage)
6. [Dataset](#-dataset)
7. [Training](#-training)
8. [Tech Stack](#️-tech-stack)
9. [Project Structure](#-project-structure)
10. [License](#-license)

---

## 🚀 Overview

**Medical Prescription OCR** (formerly **RxReader**) converts doctors' handwritten prescriptions into structured, machine-readable text with high accuracy.

### Key Capabilities
- **Accurate OCR** – Transcribes drug names, dosages, frequencies and instructions  
- **Structured Output** – Returns clean JSON with parsed prescription elements  
- **Zero-shot Classification** – Detects prescription documents vs. other medical forms  
- **Robust Performance** – Handles diverse handwriting styles and image qualities  

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Pre-trained Model** | Ready to use on the [HF Model Hub]
| 📊 **Comprehensive Dataset** | 1,000 synthetic, fully annotated images on [HF Datasets]
| 🖥️ **User-Friendly Interface** | Gradio web app for drag-and-drop testing |
| 🔄 **Gradual Augmentation** | Curriculum strategy for robust learning |
| 📈 **Production Ready** | Download script and deployment guide included |

---

## 📊 Performance

| Metric | Score | Notes |
|--------|-------|-------|
| **Character-level accuracy** | **71%** | Individual character recognition |
| **Word-level accuracy** | **84%** | Complete word recognition |
| **Processing speed** | **≈2s/img** | CPU – Apple M1 |

*Benchmarked on 100 held-out prescriptions with varied handwriting.*

---

## ⚡ Quick Start

### Prerequisites
* Python ≥ 3.8  
* ~2 GB free disk space for model files  
* *(Optional)* CUDA GPU for faster inference  

### Installation

```bash
# 1 – Clone the repo
git clone
cd medical-prescription-ocr

# 2 – Install dependencies
pip install -r requirements.txt

# 3 – Download the pre-trained model (~800 MB)
python model_download.py

# 4 – Launch the Gradio app
python app.py

```

The app will be available at **http://localhost:7860**.

---

## 🤖 Model Usage

### Basic OCR

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load model & processor
processor = DonutProcessor.from_pretrained("medical-prescription-ocr")
model = VisionEncoderDecoderModel.from_pretrained("medical-prescription-ocr")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load an image
image = Image.open("prescription.jpg").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Generate text
task_prompt = "<s_ocr>"
decoder_input_ids = processor.tokenizer(task_prompt, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=512,
    num_beams=1,
    early_stopping=True,
)

# Decode
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)

```

### Advanced – with zero-shot verification
`app.py` demonstrates automatic **zero-shot classification** (BART-based) to verify an image is a prescription before running OCR.

---

## 📚 Dataset

| Split | Samples |
|-------|---------|
| Train | 800 |
| Val   | 100 |
| Test  | 100 |

*1,000 synthetic PNG images + JSON annotations.*


---

## 🛠️ Training

The full training workflow is documented in `OCR_training.ipynb`.

| Aspect | Details |
|--------|---------|
| **Base model** | NAVER Clova Donut |
| **Framework** | PyTorch Lightning |
| **Optimizer** | AdamW with linear warm-up |
| **Strategy** | Gradual augmentation curriculum |
| **Hardware** | NVIDIA GPU (mixed precision) |

### Key Innovations
* **Gradual Augmentation** – Starts with light distortions and progressively increases difficulty
* **Smart Callbacks** – Early stopping, checkpointing, and custom schedulers
* **Memory Efficiency** – Gradient checkpointing & automatic mixed precision

---

## 🏗️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Core Framework | **PyTorch 2** | Deep-learning foundation |
| Training | **PyTorch Lightning** | Clean, reproducible loops & logging |
| Model Arch. | **Donut** (NAVER) | Document-level OCR |
| Tokenization | **SentencePiece** | Sub-word encoding |
| Augmentation | **Albumentations** | Fast, flexible image transforms |
| Classification | **BART** (Meta AI) | Zero-shot document-type detection |
| Interface | **Gradio** | Web demo |
| Hosting | **Hugging Face Hub** | Model & dataset distribution |

---

## 📁 Project Structure

```text
medical-prescription-ocr/
├── app.py               # Gradio web application
├── model_download.py    # HF model downloader
├── OCR_training.ipynb   # End-to-end training notebook
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT license
├── README.md            # This file
└── model/               # Populated after download
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── …

```


---

## ⚠️ Important Notes

* **Research use only** – The model is **not** validated for clinical workflows.
* **Synthetic data** – Trained entirely on generated prescriptions, *not* real patient data.
* **No medical advice** – Do **not** use this model to process real prescriptions.
* **Privacy** – Never upload confidential patient prescriptions to the public demo.

---

## 📄 License

Released under the **MIT License**.  
See the full text in the [`LICENSE`](LICENSE) file.

---

## 🙏 Acknowledgments

* **[NAVER Clova AI](https://github.com/clovaai/donut)** – Donut architecture
* **[Hugging Face](https://huggingface.co/)** – Model & dataset hosting
* **[Meta AI](https://github.com/facebookresearch/fairseq)** – BART zero-shot classifier
* **[IAM Handwriting DB](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)** – Inspiration for annotation schema

---

