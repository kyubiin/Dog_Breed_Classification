# Dog Breed Classification with Data & Model Enhancement

This repository contains an end-to-end pipeline for **dog breed image classification**, focusing on both **data quality improvement** and **model performance optimization**.

We construct a clean, dog-focused dataset and fine-tune several CNN architectures using modern training techniques to achieve robust performance on dog breed classification.

---

## 📖 1. Project Overview

The project consists of two main pillars:

1.  **Data Enhancement**
    * **Base:** Stanford Dogs Dataset.
    * **Augmentation:** Web-crawled images (Bing) for each breed using **YOLOv8** for detection and cropping.
    * **Dataset:** A pre-processed, merged dataset is provided via Google Drive to handle the large file size.

2.  **Model Enhancement**
    * Fine-tuning of multiple CNN backbones (`VGG16`, `ResNet50`, `GoogLeNet`, `InceptionV3`, `Xception`, `EfficientNet-B7`).
    * Application of advanced training techniques (Label Smoothing, Regularization, etc.).
    * **Grid Search:** Optimization of hyperparameters (Epochs, Batch Size, LR) for the top-performing models.

---

## 🗂️ 2. Dataset Download

Due to the large size of the dataset, it is hosted externally on Google Drive. The dataset consists of cropped images from the Stanford Dogs Dataset mixed with YOLOv8-cropped web-crawled images.

### 📥 Download Link
Please download the pre-processed dataset from the link below and extract it into the `data/` directory.

* **Google Drive Link:** [Download Dataset](https://drive.google.com/file/d/1JMsleaOJQJSCUYzJIB4NVh4S34wDBQLF/view?usp=sharing)

---

## 📂 3. Repository Structure

After downloading and extracting the dataset, your directory structure should look like this:

```text
.
├── data/                      # Dataset root directory
│   ├── n02085620-Chihuahua/   # <ID>-<BreedName> format
│   │   ├── Chihuahua_0.jpg
│   │   ├── Chihuahua_1.jpg
│   │   └── ...
│   ├── n02099601-Golden_retriever/
│   │   ├── Golden_retriever_0.jpg
│   │   └── ...
│   └── ...
├── models/                    # Training scripts & Model definitions
│   ├── train_vgg16.py
│   ├── train_resnet50.py
│   ├── train_googlenet.py
│   ├── train_inceptionv3.py
│   ├── train_xception.py
│   └── train_efficientnetb7.py
└── README.md
```

> **Note:** The `data/` folder contains subfolders for each breed (e.g., `n02085620-Chihuahua`), and inside are the image files.

-----

## 🧠 4. Models & Training Strategy

### 4.1 Architectures

We experimented with the following CNN backbones pretrained on ImageNet:

  * **Standard:** VGG16, ResNet50
  * **Inception Family:** GoogLeNet, InceptionV3, Xception
  * **EfficientNet:** EfficientNet-B7

### 4.2 Training Techniques

To improve generalization and stability:

  * **Dropout:** Randomly drops neurons to prevent overfitting.
  * **Learning Rate Scheduler:** Dynamic adjustment (e.g., step decay/cosine) for refined convergence.
  * **Label Smoothing:** Prevents the model from becoming overconfident on noisy labels.
  * **Early Stopping:** Monitors validation loss/accuracy to stop training at the optimal point.

### 4.3 Hyperparameter Search

Based on initial baselines, **GoogLeNet, Xception, and EfficientNet-B7** showed the most promise. We performed a grid search for these three models:

| Hyperparameter | Candidates |
| :--- | :--- |
| **Epochs** | `[40]` |
| **Batch Size** | `[4, 8, 16]` |
| **Learning Rate** | `[1e-5, 3e-5, 1e-4, 3e-4]` |

*Total configurations per model: 12 (1 × 3 × 4)*

-----

## 📊 5. Results

The final Top-3 Test Accuracy and Test Loss for the best-performing models:

| Model | Test Accuracy | Test Loss |
| :--- | :--- | :--- |
| **GoogLeNet** | **xx.x%** | **x.xxxx** |
| **Xception** | **xx.x%** | **x.xxxx** |
| **EfficientNet-B7** | **91.23%** | **0.3890** |

*(Note: Other backbones like VGG16 and ResNet50 achieved lower accuracy under the same constraints.)*

-----

## 🚀 6. Usage

### 6.1 Environment Setup

```bash
# Clone the repository
git clone https://github.com/kyubiin/Dog_Breed_Classification.git
cd Dog_Breed_Classification

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Data Setup

1.  Download the dataset from the [Google Drive Link](https://drive.google.com/file/d/1JMsleaOJQJSCUYzJIB4NVh4S34wDBQLF/view?usp=sharing).
2.  Unzip the file.
3.  Move the extracted breed folders into the `data/` directory so it matches the structure in Section 3.

### 6.3 Training Examples

You can train specific architectures using the scripts in the `models/` directory.

**Train Xception:**

```bash
python models/train_xception.py 
```

**Train GoogLeNet:**

```bash
python models/train_googlenet.py 
```

-----

## 🔮 7. Future Work

  * **Advanced Augmentation:** Implement techniques tailored for dogs (pose variation, color jitter).
  * **New Architectures:** Explore Vision Transformers (ViT) or hybrid ConvNet-Transformer models.
  * **Evaluation Suite:** Add scripts for confusion matrices and per-class accuracy metrics.
  * **Deployment:** Build a REST API or web demo for real-time inference.

-----

## 👏 8. Acknowledgements

  * **Data:** [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data)
  * **Tools:** [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
  * **Community:** PyTorch and the open-source Deep Learning community.
