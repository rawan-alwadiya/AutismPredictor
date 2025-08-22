# **AutismPredictor: ANN-Based Autism Predictor**

AutismPredictor is a deep learning project that applies an **Artificial Neural Network (ANN)** to predict the likelihood of Autism Spectrum Disorder (ASD) in adults using survey-based screening data.  
The project demonstrates an **end-to-end machine learning workflow** including **data exploration, preprocessing, ANN modeling, evaluation, and deployment with Streamlit**.

---

## **Demo**

- 🎥 [View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_deeplearning-artificialneuralnetworks-binaryclassification-activity-7364474655748870144-_mF7?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)  
- 🌐 [Try the App Live on Streamlit](https://autismpredictor-h6kfmers62gfwz5at9bprh.streamlit.app/)

![App Demo]()

---

## **Project Overview**

The workflow includes:  
- **Exploratory Data Analysis (EDA)** and visualization  
- **Preprocessing**: handling missing values, categorical encoding, and class imbalance correction  
- **Feature selection**: 15 behavioral, demographic, and medical history features  
- **Modeling**: ANN with multiple hidden layers, dropout, batch normalization, and regularization  
- **Evaluation**: accuracy, precision, recall, F1-score  
- **Deployment**: interactive **Streamlit web app** for real-time predictions  

---

## **Objective**

Develop and deploy a robust ANN-based classification model to predict autism screening results in adults, while addressing **class imbalance** and potential **overfitting**.

---

## **Dataset**

- **Source**: [Autism Screening on Adults Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults/data)  
- **Samples**: 704  
- **Features**: 15 selected features (10 behavioral scores, demographic & medical history inputs)  
- **Target**: Autism Screening Result (Positive = 1, Negative = 0)  

---

## **Project Workflow**

- **EDA & Visualization**: Feature distributions, missing value checks, outlier detection  
- **Preprocessing**:  
  - Filled missing values  
  - Encoded categorical features with label encoding & mappings  
  - Applied **RandomOverSampler** to handle class imbalance  
- **Modeling (ANN)**:  
  - Input layer with 15 features  
  - Hidden layers: 64 → 512 → 256 → 64 → 32  
  - ReLU activation, Batch Normalization, Dropout, L1/L2/L1L2 regularization  
  - Output: Dense(1, Sigmoid) for binary classification  
- **Training Setup**:  
  - Optimizer: Adam  
  - Loss: Binary Crossentropy  
  - EarlyStopping (patience=10, restore best weights)  
  - Batch size: 30, Epochs: 100, Validation split: 0.2  

---

## **Performance Results**

**Artificial Neural Network Classifier:**  
- **Accuracy (Train)**: 97.61%  
- **Accuracy (Test)**: 97.48%  
- **Precision**: 0.95  
- **Recall**: 1.00  
- **F1-score**: 0.98  

The model achieved **high accuracy and balanced precision-recall**, showing strong generalization without overfitting.

---

## **Project Links**

- **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/rawanalwadeya/autismpredictor-ann-based-autism-predictor)  
- **Dataset**: [Autism Screening Dataset](https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults/data)  
- **Live Streamlit App**: [Try it Now](https://autismpredictor-h6kfmers62gfwz5at9bprh.streamlit.app/)  

---

## **Tech Stack**

**Languages & Libraries**:  
- Python, Pandas, NumPy  
- Matplotlib, Seaborn  
- TensorFlow / Keras  
- Scikit-learn
- Streamlit (Deployment)  

**Techniques**:  
- ANN (deep neural network with dropout, batch normalization, regularization)  
- Oversampling (RandomOverSampler)  
- EarlyStopping
- regularization  
- Real-time web deployment with Streamlit  
