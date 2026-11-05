# AI-Based Cyberattack Detection System

## Project Overview
An AI/ML-based network traffic detection system that identifies potentially malicious network traffic using machine learning. The project uses the CIC-IDS2017 dataset and provides an interactive Streamlit dashboard for uploading CSV files and analyzing network traffic.

## Key Features
- Network traffic classification into BENIGN and ATTACK.
- Data preprocessing and exploratory data analysis.
- Feature engineering and feature importance analysis.
- Machine learning using Random Forest and LightGBM.
- Interactive Streamlit dashboard for CSV-based predictions.
- Prediction summaries, visualizations, evaluation metrics, and downloadable results.

## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- LightGBM
- Random Forest
- Streamlit
- Jupyter Notebook

## Machine Learning Model
The Streamlit application uses a trained LightGBM pipeline with median imputation and variance-based feature selection.

A Random Forest model is also retained as an alternative model.

The system performs binary classification:
- **BENIGN (0):** Normal network traffic
- **ATTACK (1):** Potentially malicious network traffic

## Dataset
The project uses CIC-IDS2017 network traffic data containing more than 2.6 million records across multiple CSV files.

The raw dataset is not included in this repository.

## Project Structure
```text
AI-Based-Cyberattack-Detection/
├── app/
│   └── app.py
├── models/
│   └── cyberattack_lightgbm_model.joblib
├── notebooks/
│   ├── 01_Dataset_Analysis.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Training.ipynb
│   └── 05_Model_Training_Backup.ipynb
├── cyberattack_rf_model.joblib
├── README.md
└── requirements.txt
```

## Run Locally

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app/app.py
```

4. Upload a CSV file containing the required network traffic features.

## Important Limitation
The model predicts BENIGN or ATTACK; it does not identify the exact attack family. Performance on unseen network traffic can differ from performance on training or randomly split data. Further validation and improvement are required before real-world deployment.

## Author
Kalyani Gaikar
