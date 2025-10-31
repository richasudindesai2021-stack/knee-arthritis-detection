
# 🦵 Detection of Knee Arthritis Using Multiple Machine Learning Models

## 📘 Overview

This project aims to **automate the diagnosis of Knee Osteoarthritis (KOA)** using deep learning techniques on X-ray images. Manual interpretation is often subjective and time-consuming — our solution leverages **multiple ML architectures** to classify arthritis into five severity levels: *Normal, Doubtful, Mild, Moderate,* and *Severe*.

The system assists healthcare professionals by providing **faster, more accurate, and consistent diagnostic predictions**, reducing human bias and clinical workload.

---

## 🎯 Objectives

* Develop a **multi-class classification model** for knee arthritis detection.
* Compare performance across **CNN, VGG16, ResNet50, and EfficientNet-B0** architectures.
* Identify the most optimal model balancing **accuracy, recall, and computational efficiency**.

---

## 🧠 Models Implemented

| Model                            | Description                                                                    | Accuracy                               | Remarks                                                      |
| -------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------- | ------------------------------------------------------------ |
| **Custom CNN**                   | 3 Conv2D layers (128–64–32 filters) to establish baseline performance.         | ~85%                                   | High training speed, lower generalization.                   |
| **VGG16 (Transfer Learning)**    | Pretrained on ImageNet, fine-tuned final layers for arthritis classification.  | ~89%                                   | Improved stability over CNN.                                 |
| **ResNet50 (Transfer Learning)** | Deep residual network with skip connections solving vanishing gradient issues. | **87% (Validation)**, **82.4% (Test)** | Top recall and F1-score for “Severe” and “Moderate” classes. |
| **EfficientNet-B0**              | Compound scaling optimization for high accuracy and low computation.           | **96–97% (Moderate class)**            | Best balance between speed and accuracy.                     |

---

## ⚙️ Methodology

### 🧩 Data Processing

* Sourced **annotated X-ray images** from open medical datasets.
* **Preprocessing:** resizing (224×224), normalization, noise removal.
* **Augmentation:** rotation, zoom, horizontal flip, brightness adjustment.

### 🧮 Model Training

* Transfer learning with **fine-tuned final layers**.
* Optimizer: *Adam*, loss: *categorical crossentropy*.
* Learning rate scheduling and early stopping to avoid overfitting.

### 📈 Evaluation Metrics

* Accuracy, F1-Score, Precision, Recall.
* Confusion matrices and classification reports per category.

---

## 🧩 Results & Insights

* **Best model:** ResNet50 with 87% validation and 82.4% test accuracy.
* **EfficientNet-B0** achieved **97% accuracy in Moderate category**, outperforming all others.
* **Key insight:** Moderate-stage arthritis is most identifiable due to distinctive bone structure changes visible on X-rays.
* **Medical relevance:** High recall in critical classes (*Moderate, Severe*) minimizes false negatives — vital for early-stage intervention.

---

## 💡 Novelty

* First comparative approach implementing **four ML architectures** for multi-class KOA classification.
* Demonstrated how **EfficientNet-B0** provides high accuracy with minimal computational overhead.
* Introduced a **lightweight CNN baseline** to assess complexity and benchmark deeper models.

---

## 🚀 Future Scope

* Integrate **Explainable AI (Grad-CAM)** to visualize heatmaps of decision areas.
* Extend to other orthopedic conditions using transfer learning.
* Deploy mobile-friendly version using **TensorFlow Lite** for real-time diagnostic assistance.

---

## 🛠️ Tech Stack

**Languages & Frameworks:** Python, TensorFlow, Keras
**Libraries:** NumPy, Pandas, OpenCV, scikit-learn, Matplotlib
**Tools:** Jupyter Notebook, PyCharm, Google Colab


