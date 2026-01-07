# COVID-19 Detection (Healthcare) --- Image Classification Using EfficientNetB0 (Transfer Learning & Fine-Tuning)

## Project Overview
This project implements an **image classification system** using **EfficientNetB0** with **transfer learning and fine-tuning**.  
A pre-trained ImageNet model is used as a feature extractor, followed by custom classification layers.  
Fine-tuning is applied by unfreezing the top layers of the base model to improve performance on the target dataset.

---

## Key Features
- Transfer learning using EfficientNetB0
- Partial fine-tuning of pre-trained layers
- Data augmentation for better generalization
- Multi-class classification using Softmax
- Performance evaluation using Accuracy and AUC

---

## Technologies Used
- Python 3
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Jupyter Notebook

---

## Model Architecture
→ Input Image
→ Data Augmentation
→ EfficientNet Preprocessing
→ EfficientNetB0 (Pre-trained)
→ Global Average Pooling
→ Dense (128, ReLU)
→ Dropout
→ Softmax Output


---

## Transfer Learning
- EfficientNetB0 loaded with ImageNet weights
- Base model initially frozen
- Custom classifier layers added on top
- Achieved ~82% validation accuracy

---

## Fine-Tuning Strategy
- Top 20 layers of EfficientNetB0 unfrozen
- Lower layers kept frozen to preserve learned features
- Learning rate reduced to avoid weight distortion
- Fine-tuning improves feature specialization

```python
for layer in base_model.layers[:-20]:
    layer.trainable = False
```
---

## Training Configuration
- Optimizer: Adam
- Learning Rate: 1e-5
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy, AUC
- Epochs: 10 (transfer learning) + 5 (fine-tuning)
---
## Results

| Training Phase    | Validation Accuracy | Validation AUC |
| ----------------- | ------------------- | -------------- |
| Transfer Learning | ~82%                | ~0.96          |
| Fine-Tuning       | ~77–80%             | ~0.94          |

