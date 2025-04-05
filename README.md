# 🫁 Lung Cancer Prediction using ML & DL

This is a two-stage lung cancer detection system built using **Machine Learning** and **Deep Learning**, developed as part of a Hackathon project. The system predicts the likelihood of lung cancer based on user symptoms, and if flagged as high risk, further classifies chest X-ray images into five possible disease categories.

## 🚀 Project Overview

We developed an end-to-end web app that:

1. **Collects basic user symptoms** (e.g., coughing, smoking, chest pain, family history)
2. **Uses a Machine Learning model** to assess high/low risk
3. If high risk, prompts the user to upload a **chest X-ray image**
4. **Deep Learning model** classifies the image into:
   - Bacterial Pneumonia
   - Viral Pneumonia
   - Tuberculosis
   - COVID-19
   - Normal Lung Cancer

## 🧠 Technologies Used

- Python
- Streamlit (for web app)
- Scikit-learn (ML model)
- TensorFlow / Keras (CNN model)
- OpenCV / PIL (image preprocessing)

---

## 🧪 Features

- Dual-layered prediction: **Symptom + X-ray based**
- Real-time web app interface
- Multi-class classification using CNN
- Clean UI with meaningful outputs
- Scalable and easily extendable

---

## 📦 Installation & Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/lung-cancer-prediction-ml-dl.git
cd lung-cancer-prediction-ml-dl
