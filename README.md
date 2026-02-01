# Disaster Management System using Deep Learning

This repository contains an academic deep learning project focused on automated
disaster classification and damage assessment using image-based data. The work
was carried out as part of undergraduate coursework with emphasis on correct
experimental methodology, model evaluation, and analysis.

The system aims to identify and categorize disaster-related scenarios such as
floods, fires, damaged infrastructure, human damage, and non-damage content.

---

## Project Overview

Rapid identification of disaster scenarios is critical for effective response
and resource allocation. This project explores the use of convolutional neural
networks to classify disaster images into meaningful damage categories.

The work involved experimenting with multiple deep learning architectures,
analyzing model behavior, and refining training strategies to achieve stable
and interpretable performance.

---

## Model Architectures

The following CNN-based architectures were explored:

- DenseNet-based model  
- ResNet-based model  

Pretrained backbones were fine-tuned on disaster-related image data. Model
performance was evaluated using confusion matrices and class-wise accuracy to
understand strengths and limitations across categories.

---

## Development Process

The system was developed through multiple experimental iterations involving
different architectures, preprocessing strategies, and training configurations.
Early experiments highlighted challenges related to class imbalance and feature
learning, which informed refinements in later stages.

The final implementation reflects these refinements and demonstrated consistent
performance across disaster classes.

---

## Key Components

- `app.py` – Inference and prediction logic  
- `dashboard.py` – Visualization and evaluation interface  
- `model.ipynb`, `model_2.ipynb` – Training and experimentation notebooks  
- `export_torchscript.py` – Model export for deployment  
- `generate_class_map.py` – Class index mapping generation  
- `class_idx_to_name.json` – Label mapping configuration  
- Confusion matrices and graphs for model evaluation  

---

## Technologies Used

- Python  
- PyTorch and TorchVision  
- Jupyter Notebook  
- NumPy, Matplotlib  

---

## Academic Context

This project was completed as part of undergraduate deep learning coursework.
The focus was on understanding model behavior, experimental rigor, and proper
evaluation practices rather than production deployment.

---

## Authors

Aditya Pavale, Pranav Sathwik, Shaman Vidyananda

Computer Science Engineering  
Amrita Vishwa Vidyapeetham, Bengaluru
