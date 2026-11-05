
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CyberShield | AI Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    .stApp {
        background-color: #0b1220;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111e33;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 4px;
    }

    .sub-title {
        color: #cbd5e1;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: linear-gradient(145deg, #192b45, #122039);
        border: 1px solid #315071;
        border-radius: 14px;
        padding: 22px;
        min-height: 150px;
    }

    .metric-label {
        color: #cbd5e1;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        margin-top: 12px;
    }

    .metric-note {
        color: #cbd5e1;
        font-size: 12px;
        margin-top: 8px;
    }

    .section-title {
        color: #ffffff;
        font-size: 23px;
        font-weight: 750;
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .info-box {
        background-color: #10213a;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 15px;
        color: #e2e8f0;
        margin: 12px 0;
    }

    .footer {
        text-align: center;
        color: #cbd5e1;
        font-size: 12px;
        padding: 25px 0 10px 0;
    }

    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span {
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #f1f5f9;
    }

    h1, h2, h3, h4, h5 {
        color: #ffffff !important;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
    }

    hr {
        border-color: #26364d;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "cyberattack_lightgbm_model.joblib"
)


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    loaded_model = joblib.load(MODEL_PATH)

    if not hasattr(loaded_model, "feature_names_in_"):
        raise ValueError(
            "The model does not contain feature names."
        )

    return loaded_model


try:

    model = load_model()

    features = list(model.feature_names_in_)

    classifier = model.named_steps["classifier"]

    model_classes = list(classifier.classes_)

    if 0 not in model_classes or 1 not in model_classes:
        raise ValueError(
            "The model must support binary classes 0 and 1."
        )

    attack_class_index = model_classes.index(1)

except Exception as error:

    st.error(f"Unable to load the LightGBM model: {error}")
    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def create_metric_card(title, value, note):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_risk_level(attack_percentage):

    if attack_percentage >= 20:
        return "HIGH", "🔴"

    elif attack_percentage >= 5:
        return "MEDIUM", "🟠"

    return "LOW", "🟢"


def get_recommendations(attack_percentage, attack_count):

    recommendations = []

    if attack_count == 0:

        recommendations.append(
            "No records were classified as attacks. "
            "Continue monitoring network traffic."
        )

    elif attack_percentage >= 20:

        recommendations.extend([
            "A high proportion of records was flagged. "
            "Prioritize manual security investigation.",

            "Review flagged traffic against firewall and IDS/IPS logs.",

            "Investigate unusual source and destination IP addresses "
            "and network ports, if available.",

            "Review access controls and monitor suspicious connections."
        ])

    elif attack_percentage >= 5:

        recommendations.extend([
            "Some network records were flagged as potential attacks.",

            "Review flagged records and verify them against security logs.",

            "Monitor unusual connection patterns and network activity."
        ])

    else:

        recommendations.extend([
            "A relatively small proportion of records was flagged.",

            "Continue regular network monitoring.",

            "Investigate individual alerts before taking security action."
        ])

    return recommendations


def convert_label_to_binary(label_series):

    numeric_labels = pd.to_numeric(
        label_series,
        errors="coerce"
    )

    if numeric_labels.notna().all():

        unique_values = set(numeric_labels.unique())

        if unique_values.issubset({0, 1}):
            return numeric_labels.astype(int)

    text_labels = (
        label_series.astype(str)
        .str.strip()
        .str.upper()
    )

    return (text_labels != "BENIGN").astype(int)


def find_timestamp_column(data):

    possible_names = [
        "Timestamp",
        "timestamp",
        "Date",
        "date",
        "Datetime",
        "datetime",
        "Flow Start",
        "Flow_Start"
    ]

    for column in possible_names:

        if column in data.columns:
            return column

    return None


def plot_traffic_distribution(benign_count, attack_count):

    fig, ax = plt.subplots(figsize=(9, 3.5))

    categories = ["BENIGN", "ATTACK"]

    values = [
        benign_count,
        attack_count
    ]

    bars = ax.barh(
        categories,
        values,
        height=0.48
    )

    ax.set_title(
        "Network Traffic Classification",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Number of Records")
    ax.set_ylabel("Traffic Category")

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):

        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {value:,}",
            va="center",
            fontsize=10
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h2 style="color:#ffffff;">🛡️ CyberShield</h2>
        <p style="color:#cbd5e1;">
            AI-POWERED SECURITY ANALYTICS
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** LightGBM")
    st.write("**Task:** Binary Classification")
    st.write("**Classes:** BENIGN / ATTACK")
    st.write(f"**Input Features:** {len(features)}")

    st.divider()

    st.subheader("Application")

    st.write("🌐 Network Traffic Detection")
    st.write("📂 CSV-based Prediction")
    st.write("🎯 Prediction Confidence")
    st.write("🚦 Traffic Alert Assessment")
    st.write("📊 Model Performance Evaluation")
    st.write("🛡️ Security Recommendations")

    st.divider()

    st.caption(
        "CyberShield is a machine-learning demonstration "
        "for network traffic classification."
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ CyberShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'AI-Based Cyberattack Detection System | '
    'Network Traffic Classification'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
        Upload a network traffic CSV file to classify records
        as BENIGN or potential ATTACK using the trained LightGBM model.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CSV UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">📂 Upload Network Traffic Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV containing the 78 required network traffic features."
)

if uploaded_file is None:

    st.info(
        "Please upload a CSV file to start network traffic detection."
    )

    st.markdown(
        """
        **Before uploading:**

        - The CSV should contain the 78 model input features.
        - The `Label` column is optional.
        - The model predicts BENIGN or ATTACK.
        - Missing numerical values are handled by the model pipeline.
        """
    )

    st.stop()


# =========================================================
# READ DATASET
# =========================================================

try:

    uploaded_file.seek(0)

    data = pd.read_csv(
        uploaded_file,
        low_memory=False
    )

    data.columns = (
        data.columns.astype(str).str.strip()
    )

except pd.errors.EmptyDataError:

    st.error("The uploaded CSV file is empty.")
    st.stop()

except Exception as error:

    st.error(f"Unable to read the uploaded CSV: {error}")
    st.stop()


if data.empty:

    st.error("The uploaded CSV file contains no records.")
    st.stop()


st.success(
    f"Uploaded successfully: {uploaded_file.name}"
)


# =========================================================
# FEATURE VALIDATION
# =========================================================

missing_features = [
    column
    for column in features
    if column not in data.columns
]

if missing_features:

    st.error(
        f"Your CSV is missing {len(missing_features)} required features."
    )

    with st.expander("View missing feature names"):

        st.write(missing_features)

    st.warning(
        "Please upload a CSV containing the required model features."
    )

    st.stop()


# =========================================================
# PREPARE INPUT DATA
# =========================================================

X = data[features].copy()

for column in X.columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# =========================================================
# MODEL PREDICTION
# =========================================================

try:

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    attack_probabilities = (
        probabilities[:, attack_class_index]
    )

    prediction_confidence = np.max(
        probabilities,
        axis=1
    )

except Exception as error:

    st.error(
        f"Prediction failed. Please check your CSV data. Error: {error}"
    )

    st.stop()


# =========================================================
# RESULTS DATAFRAME
# =========================================================

results = data.copy()

results["Prediction"] = np.where(
    predictions == 1,
    "ATTACK",
    "BENIGN"
)

results["Attack_Probability"] = (
    attack_probabilities * 100
).round(2)

results["Prediction_Confidence"] = (
    prediction_confidence * 100
).round(2)


# =========================================================
# SUMMARY CALCULATIONS
# =========================================================

total_records = len(results)

attack_count = int(
    np.sum(predictions == 1)
)

benign_count = int(
    np.sum(predictions == 0)
)

attack_percentage = (
    attack_count / total_records
) * 100

benign_percentage = (
    benign_count / total_records
) * 100

average_confidence = (
    prediction_confidence.mean() * 100
)

average_attack_probability = (
    attack_probabilities.mean() * 100
)


# =========================================================
# DETECTION OVERVIEW
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📊 Detection Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    create_metric_card(
        "Total Network Records",
        f"{total_records:,}",
        "Records analyzed"
    )

with col2:

    create_metric_card(
        "Attacks Detected",
        f"{attack_count:,}",
        f"{attack_percentage:.2f}% of uploaded traffic"
    )

with col3:

    create_metric_card(
        "Benign Traffic",
        f"{benign_count:,}",
        f"{benign_percentage:.2f}% of uploaded traffic"
    )


# =========================================================
# TRAFFIC ALERT ASSESSMENT
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🚦 Traffic Alert Assessment</div>',
    unsafe_allow_html=True
)

risk_level, risk_icon = get_risk_level(
    attack_percentage
)

st.subheader(
    f"{risk_icon} {risk_level} Flagged Traffic Volume"
)

st.write(
    f"{attack_percentage:.2f}% of uploaded records "
    "were classified as ATTACK."
)

st.caption(
    "Dashboard thresholds: Low = below 5%, "
    "Medium = 5% to below 20%, High = 20% or above. "
    "These thresholds describe flagged traffic volume, "
    "not validated real-world incident severity."
)

if risk_level == "HIGH":

    st.error(
        "High flagged traffic volume: A significant proportion "
        "of records was classified as ATTACK. Investigate the "
        "flagged records using additional security evidence."
    )

elif risk_level == "MEDIUM":

    st.warning(
        "Medium flagged traffic volume: Some network records "
        "were classified as ATTACK. Review the flagged records."
    )

else:

    st.success(
        "Low flagged traffic volume: A relatively small proportion "
        "of records was classified as ATTACK. Continue monitoring."
    )


# =========================================================
# TRAFFIC DISTRIBUTION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Traffic Distribution</div>',
    unsafe_allow_html=True
)

chart_col, details_col = st.columns([2, 1])

with chart_col:

    fig = plot_traffic_distribution(
        benign_count,
        attack_count
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

with details_col:

    st.metric(
        "Attack Traffic",
        f"{attack_percentage:.2f}%"
    )

    st.metric(
        "Benign Traffic",
        f"{benign_percentage:.2f}%"
    )

    st.metric(
        "Total Records",
        f"{total_records:,}"
    )


# =========================================================
# PREDICTION CONFIDENCE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🎯 Prediction Confidence</div>',
    unsafe_allow_html=True
)

confidence_col1, confidence_col2 = st.columns(2)

with confidence_col1:

    st.metric(
        "Average Prediction Confidence",
        f"{average_confidence:.2f}%"
    )

with confidence_col2:

    st.metric(
        "Average ATTACK Probability",
        f"{average_attack_probability:.2f}%"
    )

st.caption(
    "These values are model-predicted probabilities. "
    "They are not guarantees that a prediction is correct "
    "and may not be calibrated confidence scores."
)


# =========================================================
# SECURITY RECOMMENDATIONS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🛡️ Security Recommendations</div>',
    unsafe_allow_html=True
)

recommendations = get_recommendations(
    attack_percentage,
    attack_count
)

for index, recommendation in enumerate(
    recommendations,
    start=1
):

    st.markdown(
        f"**{index}.** {recommendation}"
    )


# =========================================================
# TIME-BASED TREND ANALYSIS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🕒 Network Traffic Trend Analysis</div>',
    unsafe_allow_html=True
)

timestamp_column = find_timestamp_column(data)

if timestamp_column:

    try:

        trend_data = results.copy()

        trend_data[timestamp_column] = pd.to_datetime(
            trend_data[timestamp_column],
            errors="coerce"
        )

        trend_data = trend_data.dropna(
            subset=[timestamp_column]
        )

        if not trend_data.empty:

            trend_data["Date"] = (
                trend_data[timestamp_column].dt.floor("h")
            )

            trend_summary = (
                trend_data
                .groupby(["Date", "Prediction"])
                .size()
                .unstack(fill_value=0)
            )

            st.line_chart(
                trend_summary,
                use_container_width=True
            )

        else:

            st.info(
                "The timestamp column was found, but it "
                "does not contain valid date-time values."
            )

    except Exception:

        st.warning(
            "Unable to generate time-based trend analysis "
            "from this CSV."
        )

else:

    st.info(
        "No recognizable timestamp column was found. "
        "Add a timestamp/date-time column to your CSV "
        "to enable time-based trend analysis."
    )


# =========================================================
# MODEL PERFORMANCE EVALUATION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📐 Model Performance Evaluation</div>',
    unsafe_allow_html=True
)

if "Label" in data.columns:

    actual_labels = convert_label_to_binary(
        data["Label"]
    )

    predicted_labels = pd.Series(
        predictions,
        index=data.index
    )

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    st.metric(
        "Prediction Accuracy on Uploaded Labels",
        f"{accuracy * 100:.2f}%"
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1]
    )

    # Confusion matrix values

    tn, fp, fn, tp = matrix.ravel()

    # Attack precision

    attack_precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    # Attack recall

    attack_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    # Attack F1-score

    attack_f1 = (
        2 * attack_precision * attack_recall /
        (attack_precision + attack_recall)
        if (attack_precision + attack_recall) > 0
        else 0
    )

    # Display confusion matrix

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual BENIGN", "Actual ATTACK"],
        columns=["Predicted BENIGN", "Predicted ATTACK"]
    )

    # Additional attack detection metrics

    st.markdown("#### Attack Detection Details")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("True Positives", f"{tp:,}")
    m2.metric("False Positives", f"{fp:,}")
    m3.metric("Missed Attacks", f"{fn:,}")
    m4.metric(
        "Attack Precision",
        f"{attack_precision * 100:.2f}%"
    )

    m5, m6 = st.columns(2)

    m5.metric(
        "Attack Recall",
        f"{attack_recall * 100:.2f}%"
    )

    m6.metric(
        "Attack F1-Score",
        f"{attack_f1 * 100:.2f}%"
    )

    # Classification report

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose().round(4)

    # Display evaluation tables

    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:

        st.markdown("#### Confusion Matrix")

        st.dataframe(
            matrix_df,
            use_container_width=True
        )

    with eval_col2:

        st.markdown("#### Classification Report")

        st.dataframe(
            report_df,
            use_container_width=True
        )

    st.warning(
        "Important evaluation note: These metrics compare predictions "
        "with the Label column in the uploaded CSV. If these records "
        "were used during model training or model selection, this is "
        "not an independent test-set evaluation."
    )

else:

    st.info(
        "No Label column was found. Prediction accuracy and "
        "the classification report cannot be calculated. "
        "The application can still classify the uploaded records."
    )


# =========================================================
# PREDICTION RESULTS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📋 Prediction Results</div>',
    unsafe_allow_html=True
)

filter_option = st.selectbox(
    "Filter prediction results",
    [
        "All Records",
        "ATTACK Only",
        "BENIGN Only"
    ]
)

if filter_option == "ATTACK Only":

    display_data = results[
        results["Prediction"] == "ATTACK"
    ]

elif filter_option == "BENIGN Only":

    display_data = results[
        results["Prediction"] == "BENIGN"
    ]

else:

    display_data = results


st.write(
    f"Showing {len(display_data):,} records."
)

st.dataframe(
    display_data.head(1000),
    use_container_width=True,
    height=400
)

if len(display_data) > 1000:

    st.caption(
        "Only the first 1,000 filtered records are displayed. "
        "Download the report to view all records."
    )


# =========================================================
# DOWNLOAD REPORT
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">⬇️ Download Detection Report</div>',
    unsafe_allow_html=True
)

csv_report = results.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Complete Prediction Report",
    data=csv_report,
    file_name="CyberShield_Detection_Report.csv",
    mime="text/csv",
    use_container_width=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🛡️ CyberShield | AI-Based Cyberattack Detection System
        <br>
        Powered by Python, Scikit-learn and LightGBM
        <br>
        For educational and analytical purposes.
        Predictions should be verified using independent
        security evidence before taking action.
    </div>
    """,
    unsafe_allow_html=True
)
