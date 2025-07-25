# Fine-Tuned Resnet18 ML-Leaf-Health-Detector using Plant Pathology 2021

## Overview
This project builds a **deep learning model** capable of classifying leaf images into various disease categories using a Plant Pathology 2021 dataset from Kaggle. Leveraging **transfer learning with ResNet18**, I fine-tuned a state-of-the-art convolutional neural network to achieve strong performance on a real-world, multi-class image classification problem.

I used official documentation and YouTube tutorials to guide my understanding of PyTorch and computer vision workflows. The data pipeline, training loop, and evaluation metrics were developed with careful adaptation, reflecting my growing ability to customize and optimize machine learning systems.

---

## Background
I hold certifications in AI fundamentals, including the Microsoft AI-900 and an introductory AI course, which provided me with a solid foundation in artificial intelligence concepts. This project builds on that knowledge by applying core AI and deep learning techniques to a real-world computer vision task—classifying plant leaf diseases using convolutional neural networks.

---

## Key Features
- **Transfer Learning with ResNet18**  
  Fine-tuned a pre-trained ResNet18 model (trained on ImageNet) by replacing the final fully connected layer with a custom layer tailored to the leaf disease classes.

- **Data Preprocessing & Augmentation**  
  Applied data augmentation techniques such as random rotations, horizontal flips, and normalization to increase generalization.

- **Label Encoding**  
  Transformed string-based labels into numeric classes using `sklearn`'s `LabelEncoder`.

- **Training & Validation Loops**  
  Implemented custom PyTorch `train_one_epoch` and `validate_one_epoch` functions to track loss, accuracy, and dynamically save the best-performing model.

- **Metrics & Visualization**  
  - Plotted training vs. validation accuracy and loss curves across epochs.  
  - Generated a **confusion matrix** and **classification report** to evaluate class-wise performance.  
  - Visualized model predictions on sample images to interpret results.

---

## Tech Stack
- **Languages/Frameworks:** Python, PyTorch, Torchvision  
- **Data Handling & Visualization:** Pandas, NumPy, Matplotlib, Seaborn  
- **Modeling:** ResNet18 (Transfer Learning)  
- **Evaluation Metrics:** Accuracy, Confusion Matrix, Classification Report

---

## Results
- Successfully trained and validated the model using a **70/30 split** of the dataset.  
- Achieved a strong baseline validation accuracy after just 1–2 epochs on CPU (with more potential on GPU).

---

## What I Learned
- **Deep understanding of transfer learning:** How pre-trained CNNs can be adapted for new datasets.  
- **End-to-end ML workflow:** From preprocessing and encoding data to training and evaluating a model.  
- **Model interpretability:** Using metrics and visualizations to analyze predictions.  
- **Efficient debugging:** Optimizing data pipelines and training loops on CPU (with future scalability to GPU).

---

## Future Improvements
- Experiment with larger models (ResNet50, EfficientNet).  
- Add mixed-precision training for faster computation.  
- Deploy the trained model via a simple web interface (Flask or FastAPI).

---

## Note on Training Runtime

During development, training was manually interrupted (KeyboardInterrupt) due to long runtime on CPU hardware. This was intentional to demonstrate the training process within a reasonable time frame.

For full training, consider running the model on GPU-enabled hardware or increasing epochs as needed. The saved model weights reflect the best checkpoint from the completed training epochs.

---

## How to Run
1. **Clone this repository**  
   ```bash
   git clone https://github.com/pranavvadd/plant-pathology-resnet.git
   cd plant-pathology-resnet
