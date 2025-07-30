
# 🌱 Plant Counting Using Transfer Learning

This project implements a plant counting model using **transfer learning** with **MobileNetV2** and **convolutional neural networks (CNN)**. The model predicts plant counts from aerial images of rice/cotton seed plants stored as TFRecords.

## 📌 Project Highlights
- **Input data:** Aerial images of rice and cotton plants (TFRecords format)
- **Model architecture:** MobileNetV2 as the base model (transfer learning) + custom dense layers for regression
- **Pipeline:** Efficient TensorFlow data pipeline with batching and prefetching
- **Training:** Early stopping and learning rate reduction callbacks applied
- **Evaluation:** RMSE, R² score, and actual vs predicted scatter plot

## 🚀 How it works
1️⃣ **Load and decode TFRecord dataset containing labeled crop field aerial images. We have for:** 

    Training - 3062 samples
    Testing - 31 samples
    Validation - 57 samples
    
2️⃣ **Preprocess images:** Resize to 224×224 and normalize  
3️⃣ **Apply MobileNetV2 backbone (pre-trained on ImageNet)**  
4️⃣ **Train regression head with dense layers**  
5️⃣ **Evaluate using RMSE and R² metrics**  

## 🛠 Tools & Frameworks
- TensorFlow
- MobileNetV2 (transfer learning)
- Matplotlib (visualization)

## ⚙ Example Code Snippet
```python
base_model = applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='relu')
])

```

## 📈 Results
- **Test RMSE:** 8.26
- **Test R² score:** 0.71    

## 🌾 Future Work
- Extend the model for other crop types
- Incorporate multispectral image data
- Deploy as part of precision agriculture monitoring tools
- Crop disruption detection

## 📂 Output
✅ The trained model is saved as:
```
model.keras
```
✅ Includes actual vs predicted scatter plot visualization.

## 💻 Author
**Aman Yadav**  
[LinkedIn](https://www.linkedin.com/in/aman-yadav-py/) | [GitHub](https://github.com/Aman-Yadav-PY)
