
# 🛡️ CyberShield AI – AI-Based Cyberattack Detection System

## Project Overview
CyberShield AI is a machine learning-based cyberattack detection system designed to classify known network attacks and identify potentially suspicious network traffic.

The system combines supervised and unsupervised machine learning techniques to improve network traffic monitoring.

## Key Features
- Multi-class cyberattack classification using LightGBM.
- Anomaly detection using Isolation Forest.
- Detection of BENIGN and known attack categories.
- Identification of potentially suspicious traffic.
- CSV file upload for network traffic analysis.
- Interactive Streamlit dashboard.
- Security recommendations based on detection results.

## Machine Learning Models
1. **LightGBM:** Classifies network traffic into trained categories.
2. **Isolation Forest:** Identifies unusual traffic patterns that may indicate suspicious activity.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- LightGBM
- Streamlit
- Matplotlib and Seaborn
- Joblib

## Dataset
The project uses network traffic data based on the CIC-IDS2017 dataset, containing approximately 2.6 million records and 78 numerical features.

## Project Structure
```text
CyberShield-AI/
├── app/
│   └── app_v2.py
├── models/
│   ├── cyberattack_multiclass_lightgbm_model.joblib
│   └── cyberattack_isolation_forest.joblib
├── notebooks/
├── results/
├── src/
├── requirements.txt
└── README.md
```

## How to Run the Application

1. Clone the repository:
   https://github.com/kalyanigaikar-2005/AI-Based-Cyberattack-Detection.git

2. Install dependencies:
   `pip install -r requirements.txt`

3. Run the application:
   `python -m streamlit run app/app_v2.py`

## Important Note
The system is a machine learning prototype. Its predictions depend on the patterns learned during training. Anomaly alerts indicate unusual traffic and do not confirm that an attack has occurred. Performance on unseen network environments may vary.
