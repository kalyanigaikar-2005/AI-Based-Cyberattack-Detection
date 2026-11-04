# 🛡️ AI-Based Cyberattack Detection System

## 📌 Project Overview

The **AI-Based Cyberattack Detection System** is a machine learning project designed to analyze network traffic and identify potentially malicious activities. It classifies network traffic into two categories: **BENIGN** and **ATTACK** using supervised machine learning algorithms.

The project combines data analysis, machine learning and an interactive Streamlit application to demonstrate how AI/ML techniques can support network security and cyberattack detection.

## 🎯 Project Objectives

- Analyze network traffic data to identify patterns associated with cyberattacks.
- Perform data cleaning, preprocessing and exploratory data analysis.
- Develop machine learning models for network traffic classification.
- Evaluate model performance using multiple evaluation metrics.
- Build an interactive application for CSV-based attack predictions.

## 📂 Dataset Information

- **Dataset:** CIC-IDS2017 network traffic dataset
- **Records:** Approximately 2.6 million
- **Features:** 78 numerical features used for model development
- **Classification:** BENIGN and ATTACK

The dataset contains network traffic records representing normal activities and different types of cyberattacks.

## 🛠️ Technologies & Libraries

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Algorithms | Random Forest, XGBoost |
| Model Storage | Joblib |
| Web Application | Streamlit |
| Development Environment | VS Code, Jupyter Notebook |

## ⚙️ Project Workflow

1. **Dataset Analysis:** Examined the eight CSV files and explored their structure, features and target labels.
2. **Data Preprocessing:** Handled missing values, duplicate records, infinite values and inconsistent labels.
3. **Exploratory Data Analysis:** Analyzed traffic distribution, attack categories and feature relationships using visualizations.
4. **Feature Engineering:** Prepared numerical features and created binary target labels for classification.
5. **Model Development:** Implemented Random Forest and XGBoost algorithms.
6. **Model Evaluation:** Compared model performance using accuracy, precision, recall, F1-score and day-wise validation.
7. **Application Development:** Built an interactive Streamlit application for uploading CSV files and generating predictions.

## 🤖 Machine Learning Models

Two supervised machine learning algorithms were implemented:

- **Random Forest Classifier:** Used to classify network traffic based on multiple decision trees.
- **XGBoost Classifier:** Used as a gradient boosting algorithm for network traffic classification.

The models were evaluated using random train-test splitting and separate day-wise validation to examine their ability to generalize to different network traffic data.

## 📊 Model Evaluation

The final Random Forest model achieved approximately **99.92% accuracy on the random test split**.

Additional day-wise testing showed a significant performance drop, highlighting challenges in detecting attack types that were not represented in the training data.

This demonstrates the importance of evaluating machine learning models beyond accuracy and testing their generalization capabilities.

## 🌐 Streamlit Application

The project includes an interactive Streamlit application with the following features:

- Upload network traffic data in CSV format.
- Automatically preprocess uploaded features.
- Predict BENIGN or ATTACK traffic.
- View total records, attack predictions and benign predictions.
- Visualize prediction summaries.
- View evaluation metrics when actual labels are available.
- Download prediction results.

## 🚀 How to Run the Project

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/AI-Based-Cyberattack-Detection.git
```

**2. Navigate to the project directory**

```bash
cd AI-Based-Cyberattack-Detection
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app/app.py
```

**5. Open the application**

The application will open in your browser, usually at:

`http://localhost:8501`

## ⚠️ Limitations

- The application performs CSV-based predictions and does not monitor live network traffic.
- Model performance may decrease when evaluated on previously unseen attack types.
- The system provides binary classification (BENIGN or ATTACK), not individual attack-category predictions.
- Model predictions should not be treated as a replacement for professional cybersecurity monitoring systems.

## 🔮 Future Improvements

- Improve model generalization on unseen attack types.
- Integrate Explainable AI (XAI) techniques.
- Explore anomaly detection methods.
- Extend the system towards real-time network traffic monitoring.

## 👩‍💻 Author

**Kalyani Gaikar**

Data Science & Analytics | Machine Learning | Python | SQL | Power BI

[GitHub](https://github.com/kalyanigaikar-2005) | [LinkedIn](https://linkedin.com/in/kalyanigaikar-586517401/)

---

*This project was developed for learning and demonstrating the application of machine learning techniques in network traffic classification and cyberattack detection.*