
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CyberShield | AI Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. PROFESSIONAL DASHBOARD CSS
# ==================================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background-color: #0B1220;
    color: #E5E7EB;
}

/* Remove white header background */

[data-testid="stHeader"] {
    background-color: #0B1220;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #111C30;
    border-right: 1px solid #263750;
}

[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

/* Main title */

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 8px;
}

.subtitle {
    color: #A9BAD0;
    font-size: 15px;
    margin-bottom: 25px;
}

/* Section headings */

.section-title {
    color: #F8FAFC;
    font-size: 22px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 16px;
}

/* Metric cards */

.metric-card {
    background: linear-gradient(135deg, #17263D, #132039);
    border: 1px solid #355071;
    border-radius: 14px;
    padding: 22px;
    min-height: 125px;
}

.metric-label {
    color: #B8C9DD;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #F8FAFC;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

.metric-note {
    color: #A9BAD0;
    font-size: 12px;
    margin-top: 5px;
}

/* Upload section */

[data-testid="stFileUploader"] {
    background-color: #17263D;
    border: 1px dashed #3B82F6;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p {
    color: #E5E7EB !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #1B2D48 !important;
    border: 1px dashed #60A5FA !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #F8FAFC !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background-color: #2563EB !important;
    color: white !important;
    border: none !important;
}

/* Buttons */

div.stButton > button {
    background: linear-gradient(90deg, #2563EB, #0EA5E9);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 12px 22px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #1D4ED8, #0284C7);
    color: white;
    border: 1px solid #60A5FA;
}

/* Download button */

.stDownloadButton button {
    background-color: #17263D;
    color: #F8FAFC;
    border: 1px solid #3B82F6;
    border-radius: 8px;
}

/* Expanders */

[data-testid="stExpander"] {
    background-color: #132039;
    border: 1px solid #2B4261;
    border-radius: 10px;
}

[data-testid="stExpander"] summary {
    color: #F8FAFC !important;
}

[data-testid="stExpander"] summary p {
    color: #F8FAFC !important;
}

/* Dataframes */

[data-testid="stDataFrame"] {
    border: 1px solid #2B4261;
    border-radius: 8px;
}

/* Select box */

[data-testid="stSelectbox"] label {
    color: #E5E7EB !important;
}

/* Dividers */

hr {
    border-color: #263750;
}

/* Footer */

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 12px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. LOAD TRAINED MODEL
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR / "models" / "cyberattack_rf_model.joblib"
)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]
medians = artifact["medians"]

# ==================================================
# 4. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 🛡️ CyberShield")

    st.caption("AI-POWERED SECURITY ANALYTICS")

    st.divider()

    st.markdown("### Model Information")

    st.write("**Algorithm:** Random Forest")
    st.write("**Task:** Binary Classification")
    st.write("**Classes:** BENIGN / ATTACK")
    st.write("**Input Features:**", len(features))

    st.divider()

    st.markdown("### Application")

    st.write("Network Traffic Detection")
    st.write("CSV-based Prediction")

    st.divider()

    st.caption("AI-Based Cyberattack Detection System")

# ==================================================
# 5. MAIN HEADER
# ==================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ Cyberattack Detection Center'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent network traffic analysis powered by Machine Learning. '
    'Identify suspicious traffic and analyze detection performance.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ==================================================
# 6. CSV FILE UPLOAD
# ==================================================

st.markdown(
    '<div class="section-title">'
    '📂 Network Traffic Analysis'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your network traffic CSV file",
    type=["csv"],
    key="network_traffic_uploader"
)

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        data.columns = data.columns.str.strip()

        st.success("Dataset uploaded successfully!")

        col_a, col_b = st.columns(2)

        col_a.write("**File:** " + uploaded_file.name)

        col_b.write(
            "**Records:** " + f"{len(data):,}"
        )

        with st.expander("👁️ Preview Uploaded Dataset"):

            st.dataframe(
                data.head(10),
                use_container_width=True
            )

        # ==================================================
        # 7. CHECK REQUIRED FEATURES
        # ==================================================

        missing_features = [
            col for col in features
            if col not in data.columns
        ]

        if missing_features:

            st.error(
                f"CSV is missing {len(missing_features)} "
                "required model features."
            )

            st.write("Missing Features:", missing_features)

        else:

            st.divider()

            if st.button("🔍 Analyze Network Traffic"):

                with st.spinner(
                    "Analyzing network traffic..."
                ):

                    X = data[features].copy()

                    X = X.apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    X = X.replace(
                        [np.inf, -np.inf],
                        np.nan
                    )

                    X = X.fillna(medians)

                    predictions = model.predict(X)

                    results = data.copy()

                    results["Prediction"] = np.where(
                        predictions == 1,
                        "ATTACK",
                        "BENIGN"
                    )

                # ==================================================
                # 8. DETECTION SUMMARY
                # ==================================================

                total_records = len(results)

                total_attacks = int(
                    (predictions == 1).sum()
                )

                total_benign = int(
                    (predictions == 0).sum()
                )

                attack_percentage = (
                    total_attacks / total_records * 100
                    if total_records else 0
                )

                benign_percentage = (
                    total_benign / total_records * 100
                    if total_records else 0
                )

                st.divider()

                st.markdown(
                    '<div class="section-title">'
                    '📊 Detection Overview'
                    '</div>',
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            TOTAL NETWORK RECORDS
                        </div>
                        <div class="metric-value">
                            {total_records:,}
                        </div>
                        <div class="metric-note">
                            Records analyzed
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            ATTACKS DETECTED
                        </div>
                        <div class="metric-value">
                            {total_attacks:,}
                        </div>
                        <div class="metric-note">
                            {attack_percentage:.2f}% of traffic
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with c3:

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            BENIGN TRAFFIC
                        </div>
                        <div class="metric-value">
                            {total_benign:,}
                        </div>
                        <div class="metric-note">
                            {benign_percentage:.2f}% of traffic
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ==================================================
                # 9. TRAFFIC DISTRIBUTION
                # ==================================================

                st.markdown(
                    '<div class="section-title">'
                    '📈 Traffic Distribution'
                    '</div>',
                    unsafe_allow_html=True
                )

                chart_data = pd.DataFrame({
                    "Traffic Type": ["ATTACK", "BENIGN"],
                    "Records": [
                        total_attacks,
                        total_benign
                    ]
                })

                st.bar_chart(
                    chart_data.set_index("Traffic Type"),
                    color="#38BDF8"
                )

                # ==================================================
                # 10. MODEL PERFORMANCE
                # ==================================================

                if "Label" in data.columns:

                    actual = (
                        data["Label"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .ne("BENIGN")
                        .astype(int)
                    )

                    accuracy = accuracy_score(
                        actual,
                        predictions
                    )

                    st.divider()

                    st.markdown(
                        '<div class="section-title">'
                        '🎯 Model Performance Evaluation'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.metric(
                        "Prediction Accuracy",
                        f"{accuracy:.2%}"
                    )

                    cm = confusion_matrix(
                        actual,
                        predictions,
                        labels=[0, 1]
                    )

                    report = classification_report(
                        actual,
                        predictions,
                        labels=[0, 1],
                        target_names=["BENIGN", "ATTACK"],
                        output_dict=True,
                        zero_division=0
                    )

                    report_df = pd.DataFrame(
                        report
                    ).transpose()

                    e1, e2 = st.columns(2)

                    with e1:

                        st.markdown("**Confusion Matrix**")

                        cm_df = pd.DataFrame(
                            cm,
                            index=[
                                "Actual BENIGN",
                                "Actual ATTACK"
                            ],
                            columns=[
                                "Predicted BENIGN",
                                "Predicted ATTACK"
                            ]
                        )

                        st.dataframe(
                            cm_df,
                            use_container_width=True
                        )

                    with e2:

                        st.markdown(
                            "**Classification Report**"
                        )

                        st.dataframe(
                            report_df.round(4),
                            use_container_width=True
                        )

                    st.info(
                        "Accuracy is calculated by comparing predictions "
                        "with the original labels in the uploaded CSV. "
                        "This is not an independent evaluation if the "
                        "records were included in model training."
                    )

                else:

                    st.warning(
                        "Original Label column not found. "
                        "Performance metrics cannot be calculated."
                    )

                # ==================================================
                # 11. INDIVIDUAL PREDICTIONS
                # ==================================================

                st.divider()

                st.markdown(
                    '<div class="section-title">'
                    '📋 Individual Prediction Results'
                    '</div>',
                    unsafe_allow_html=True
                )

                filter_type = st.selectbox(
                    "Filter Predictions",
                    [
                        "All Traffic",
                        "ATTACK Only",
                        "BENIGN Only"
                    ]
                )

                if filter_type == "ATTACK Only":

                    display_results = results[
                        results["Prediction"] == "ATTACK"
                    ]

                elif filter_type == "BENIGN Only":

                    display_results = results[
                        results["Prediction"] == "BENIGN"
                    ]

                else:

                    display_results = results

                st.dataframe(
                    display_results.head(500),
                    use_container_width=True
                )

                # ==================================================
                # 12. DOWNLOAD PREDICTION REPORT
                # ==================================================

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Complete Prediction Report",
                    data=csv,
                    file_name="cyberattack_detection_results.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(f"An error occurred: {e}")

else:

    st.info(
        "Upload a network traffic CSV file to begin detection."
    )

    st.markdown("""
    ### How it works

    1. Upload network traffic data.
    2. Validate required model features.
    3. Run the trained Random Forest model.
    4. Review detection results and performance.
    5. Download the prediction report.
    """)

# ==================================================
# 13. FOOTER
# ==================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '🛡️ AI-Based Cyberattack Detection System '
    '| Machine Learning Project '
    '| CSV-Based Network Traffic Analysis'
    '</div>',
    unsafe_allow_html=True
)

