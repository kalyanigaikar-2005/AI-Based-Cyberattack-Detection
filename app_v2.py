
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from datetime import datetime


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CyberShield AI | Cyberattack Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

##======CUSTOM CSS=========

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #0B1220;
    color: #E2E8F0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111C2E;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #E2E8F0 !important;
}

/* Main headings */
h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 700;
}

/* Normal text */
p, label, li {
    color: #CBD5E1;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #17263B;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #334155;
}

[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
}

[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
}

/* Upload section */
[data-testid="stFileUploader"] {
    background-color: #17263B;
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 15px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background-color: #0284C7;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #0369A1;
    color: white;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
}

/* Cards */
.metric-card,
.info-card,
.warning-card,
.danger-card,
.success-card {
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    color: #F8FAFC;
}

.info-card {
    background-color: #172B45;
    border-left: 4px solid #38BDF8;
}

.warning-card {
    background-color: #3B2E19;
    border-left: 4px solid #F59E0B;
}

.danger-card {
    background-color: #3B2029;
    border-left: 4px solid #EF4444;
}

.success-card {
    background-color: #163529;
    border-left: 4px solid #22C55E;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MULTICLASS_MODEL_PATH = (
    PROJECT_DIR / "models" /
    "cyberattack_multiclass_lightgbm_model.joblib"
)

ISOLATION_MODEL_PATH = (
    PROJECT_DIR / "models" /
    "cyberattack_isolation_forest.joblib"
)


# ============================================================
# 4. LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    if not MULTICLASS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Multiclass model not found: {MULTICLASS_MODEL_PATH}"
        )

    if not ISOLATION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Isolation Forest model not found: {ISOLATION_MODEL_PATH}"
        )

    multiclass_artifact = joblib.load(
        MULTICLASS_MODEL_PATH
    )

    isolation_artifact = joblib.load(
        ISOLATION_MODEL_PATH
    )

    multiclass_model = multiclass_artifact["model"]

    multiclass_features = list(
        multiclass_artifact["feature_columns"]
    )

    isolation_model = isolation_artifact["model"]

    isolation_features = list(
        isolation_artifact["feature_columns"]
    )

    if multiclass_features != isolation_features:
        raise ValueError(
            "The feature columns of the two models do not match."
        )

    return (
        multiclass_model,
        isolation_model,
        multiclass_features
    )


try:

    (
        multiclass_model,
        isolation_model,
        FEATURES
    ) = load_models()

    MODEL_CLASSES = list(multiclass_model.classes_)

except Exception as error:

    st.error(f"Model loading failed: {error}")

    st.stop()


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.title("🛡️ CyberShield AI")

st.sidebar.caption(
    "Intelligent Cyberattack Detection System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Traffic Detection",
        "Security Recommendations",
        "About Models"
    ],
    index=1
)

st.sidebar.divider()

st.sidebar.markdown("### Model Status")

st.sidebar.success("Multiclass LightGBM: Loaded")

st.sidebar.success("Isolation Forest: Loaded")

st.sidebar.caption(
    f"Total model features: {len(FEATURES)}"
)

st.sidebar.divider()

st.sidebar.caption(
    "CyberShield AI | Network Security Analytics"
)


# ============================================================
# 6. MAIN HEADER
# ============================================================

st.title("🛡️ CyberShield AI")

st.markdown(
    "### AI-Based Cyberattack Detection and Anomaly Monitoring"
)

st.markdown(
    """
    Analyze network traffic, identify known cyberattacks,
    and flag unusual network behaviour using machine learning.
    """
)

st.divider()


# ============================================================
# 7. HELPER FUNCTION: DATA PREPROCESSING
# ============================================================

def prepare_features(data):

    data = data.copy()

    # Remove extra spaces from column names
    data.columns = data.columns.str.strip()

    # Check for duplicate columns
    if data.columns.duplicated().any():
        raise ValueError(
            "Your CSV contains duplicate column names. "
            "Please correct them before uploading."
        )

    # Check missing model features
    missing_features = [
        feature for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing {len(missing_features)} required features. "
            f"Examples: {missing_features[:10]}"
        )

    # Select features in the exact training order
    X = data.reindex(columns=FEATURES).copy()

    # Convert all features to numeric
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values
    X = X.fillna(0)

    return X


# ============================================================
# 8. HELPER FUNCTION: COMBINED PREDICTION
# ============================================================

def detect_attacks(X):

    # Multiclass LightGBM predictions
    known_predictions = multiclass_model.predict(X)

    # Prediction probabilities
    probabilities = multiclass_model.predict_proba(X)

    prediction_confidence = np.max(
        probabilities,
        axis=1
    ) * 100

    # Isolation Forest predictions
    # +1 = Normal
    # -1 = Anomaly

    isolation_predictions = isolation_model.predict(X)

    # Isolation Forest decision scores
    anomaly_scores = isolation_model.decision_function(X)

    # Convert results into a DataFrame
    results = pd.DataFrame(index=X.index)

    results["Known_Attack_Prediction"] = known_predictions

    results["LightGBM_Confidence"] = prediction_confidence

    results["Anomaly_Status"] = np.where(
        isolation_predictions == -1,
        "Anomaly",
        "Normal"
    )

    results["Anomaly_Score"] = anomaly_scores

    # Combined final status
    benign_mask = (
        results["Known_Attack_Prediction"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("BENIGN")
    )

    anomaly_mask = isolation_predictions == -1

    results["Final_Status"] = np.where(
        ~benign_mask,
        results["Known_Attack_Prediction"].astype(str),
        np.where(
            anomaly_mask,
            "SUSPICIOUS / POTENTIALLY UNKNOWN",
            "BENIGN"
        )
    )

    results["Risk_Level"] = results["Final_Status"].apply(
        lambda value:
        "High" if value not in [
            "BENIGN",
            "SUSPICIOUS / POTENTIALLY UNKNOWN"
        ]
        else (
            "Medium"
            if value == "SUSPICIOUS / POTENTIALLY UNKNOWN"
            else "Low"
        )
    )

    return results


# ============================================================
# 9. HELPER FUNCTION: SECURITY RECOMMENDATIONS
# ============================================================

def get_recommendation(status):

    recommendations = {

        "BENIGN": (
            "No known attack detected. Continue monitoring "
            "network traffic and maintain security updates."
        ),

        "DDoS": (
            "Consider rate limiting, traffic filtering, "
            "DDoS protection services, and firewall rules."
        ),

        "PortScan": (
            "Review exposed ports, restrict unnecessary "
            "services, and monitor repeated connection attempts."
        ),

        "DoS Hulk": (
            "Investigate high-volume HTTP requests and "
            "consider web application firewall protection."
        ),

        "DoS GoldenEye": (
            "Inspect suspicious HTTP traffic and apply "
            "appropriate rate-limiting policies."
        ),

        "DoS slowloris": (
            "Review incomplete HTTP connections and configure "
            "connection timeouts."
        ),

        "DoS Slowhttptest": (
            "Investigate slow HTTP connection behaviour and "
            "strengthen server connection limits."
        ),

        "FTP-Patator": (
            "Review FTP login attempts, enable account lockout "
            "policies, and use secure file transfer protocols."
        ),

        "SSH-Patator": (
            "Review SSH authentication logs, disable password "
            "login where practical, and enable key-based access."
        ),

        "Bot": (
            "Investigate unusual outbound traffic, isolate "
            "affected devices if necessary, and scan endpoints."
        ),

        "Web Attack - Brute Force": (
            "Apply account lockout policies, multi-factor "
            "authentication, and login rate limiting."
        ),

        "SUSPICIOUS / POTENTIALLY UNKNOWN": (
            "Investigate this unusual traffic. Review source "
            "and destination IPs, network logs, and connection "
            "patterns before taking action."
        )
    }

    return recommendations.get(
        status,
        "Investigate the traffic and review relevant network logs."
    )


# ============================================================
# 10. DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    st.subheader("Security Overview")

    st.info(
        "Upload a network traffic CSV from the Traffic Detection "
        "page to generate the dashboard."
    )

    if "prediction_results" in st.session_state:

        results = st.session_state["prediction_results"]

        total = len(results)

        benign = (
            results["Final_Status"] == "BENIGN"
        ).sum()

        suspicious = (
            results["Final_Status"]
            == "SUSPICIOUS / POTENTIALLY UNKNOWN"
        ).sum()

        known_attacks = total - benign - suspicious

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Traffic Records",
                f"{total:,}"
            )

        with col2:

            st.metric(
                "Benign Traffic",
                f"{benign:,}"
            )

        with col3:

            st.metric(
                "Known Attacks",
                f"{known_attacks:,}"
            )

        with col4:

            st.metric(
                "Suspicious Traffic",
                f"{suspicious:,}"
            )

        st.divider()

        st.subheader("Traffic Distribution")

        distribution = (
            results["Final_Status"]
            .value_counts()
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Traffic Categories")

            st.bar_chart(distribution)

        with col2:

            st.markdown("#### Attack Categories")

            attack_data = results[
                results["Final_Status"] != "BENIGN"
            ]

            if not attack_data.empty:

                st.bar_chart(
                    attack_data["Final_Status"].value_counts()
                )

            else:

                st.success("No attacks detected in this upload.")

        st.divider()

        st.subheader("Recent Detection Results")

        st.dataframe(
            results.head(20),
            use_container_width=True
        )

    else:

        st.warning(
            "No prediction data available. "
            "Upload a CSV to view security analytics."
        )


# ============================================================
# 11. TRAFFIC DETECTION PAGE
# ============================================================

elif page == "Traffic Detection":

    st.subheader("Network Traffic Analysis")

    st.markdown(
        "Upload a network traffic CSV containing the "
        "78 features required by the trained models."
    )

    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(
                uploaded_file,
                low_memory=False
            )

            st.success("CSV uploaded successfully!")

            st.markdown("### Uploaded Dataset Overview")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Total Records",
                    f"{len(data):,}"
                )

            with col2:

                st.metric(
                    "Total Columns",
                    len(data.columns)
                )

            with col3:

                st.metric(
                    "Required Features",
                    len(FEATURES)
                )

            with st.expander("Preview Uploaded Dataset"):

                st.dataframe(
                    data.head(10),
                    use_container_width=True
                )

            st.divider()

            if st.button(
                "🚀 Analyze Network Traffic",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing network traffic using AI models..."
                ):

                    X = prepare_features(data)

                    results = detect_attacks(X)

                    # Keep original uploaded records
                    final_results = data.reset_index(drop=True).copy()

                    # Add predictions
                    for column in results.columns:

                        final_results[column] = (
                            results[column].to_numpy()
                        )

                    st.session_state[
                        "prediction_results"
                    ] = final_results

                    st.session_state[
                        "analysis_completed"
                    ] = True

                st.success(
                    "Network traffic analysis completed!"
                )

                st.rerun()

        except Exception as error:

            st.error(
                f"Unable to analyze the uploaded CSV: {error}"
            )

    if "prediction_results" in st.session_state:

        results = st.session_state[
            "prediction_results"
        ]

        st.divider()

        st.subheader("Detection Summary")

        total = len(results)

        benign = (
            results["Final_Status"] == "BENIGN"
        ).sum()

        suspicious = (
            results["Final_Status"]
            == "SUSPICIOUS / POTENTIALLY UNKNOWN"
        ).sum()

        attacks = total - benign - suspicious

        col1, col2, col3 = st.columns(3)

        col1.metric("Benign", f"{benign:,}")

        col2.metric("Known Attacks", f"{attacks:,}")

        col3.metric("Suspicious", f"{suspicious:,}")

        st.divider()

        st.subheader("Detailed Prediction Results")

        filter_option = st.selectbox(
            "Filter Results",
            [
                "All",
                "BENIGN",
                "Known Attacks",
                "Suspicious"
            ]
        )

        if filter_option == "BENIGN":

            filtered_results = results[
                results["Final_Status"] == "BENIGN"
            ]

        elif filter_option == "Known Attacks":

            filtered_results = results[
                (results["Final_Status"] != "BENIGN") &
                (
                    results["Final_Status"]
                    != "SUSPICIOUS / POTENTIALLY UNKNOWN"
                )
            ]

        elif filter_option == "Suspicious":

            filtered_results = results[
                results["Final_Status"]
                == "SUSPICIOUS / POTENTIALLY UNKNOWN"
            ]

        else:

            filtered_results = results

        st.dataframe(
            filtered_results.head(1000),
            use_container_width=True
        )

        st.caption(
            "Showing up to 1,000 rows in the preview. "
            "Download the complete results below."
        )

        csv_data = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Complete Prediction Results",
            data=csv_data,
            file_name="cybershield_detection_results.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# 12. SECURITY RECOMMENDATIONS PAGE
# ============================================================

elif page == "Security Recommendations":

    st.subheader("Security Recommendations")

    st.markdown(
        "Recommendations based on the detected traffic category."
    )

    if "prediction_results" in st.session_state:

        results = st.session_state[
            "prediction_results"
        ]

        detected_categories = (
            results["Final_Status"]
            .value_counts()
        )

        for category, count in detected_categories.items():

            recommendation = get_recommendation(category)

            if category == "BENIGN":

                st.markdown(
                    f"""
                    <div class="success-card">
                    <b>🟢 {category}</b><br>
                    Records detected: {count:,}<br><br>
                    {recommendation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif category == "SUSPICIOUS / POTENTIALLY UNKNOWN":

                st.markdown(
                    f"""
                    <div class="warning-card">
                    <b>🟠 {category}</b><br>
                    Records detected: {count:,}<br><br>
                    {recommendation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="danger-card">
                    <b>🔴 {category}</b><br>
                    Records detected: {count:,}<br><br>
                    {recommendation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:

        st.info(
            "Please upload and analyze network traffic first "
            "to generate recommendations."
        )


# ============================================================
# 13. ABOUT MODELS PAGE
# ============================================================

elif page == "About Models":

    st.subheader("Machine Learning Models")

    st.markdown("### 1. Multiclass LightGBM")

    st.markdown(
        """
        LightGBM is a gradient boosting algorithm used to
        classify network traffic into known categories.

        **Main capabilities:**
        - Identify BENIGN network traffic.
        - Recognize known cyberattack categories.
        - Generate prediction probabilities.
        - Support multiclass classification.
        """
    )

    st.markdown("### 2. Isolation Forest")

    st.markdown(
        """
        Isolation Forest is an unsupervised anomaly detection
        algorithm trained on benign network traffic.

        **Main capabilities:**
        - Identify unusual network behaviour.
        - Flag potentially unknown or suspicious traffic.
        - Detect deviations from learned normal behaviour.

        An anomaly is not automatically a confirmed cyberattack.
        Some legitimate traffic may also be flagged.
        """
    )

    st.markdown("### 3. Combined Detection Strategy")

    st.markdown(
        """
        CyberShield combines both models:

        | LightGBM | Isolation Forest | Final Result |
        |---|---|---|
        | BENIGN | Normal | BENIGN |
        | BENIGN | Anomaly | Suspicious / Potentially Unknown |
        | Known attack | Normal | Known Attack |
        | Known attack | Anomaly | Known Attack |

        Known attack predictions are retained even if
        Isolation Forest considers the traffic normal.
        """
    )

    st.markdown("### Model Information")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "LightGBM Classes",
            len(MODEL_CLASSES)
        )

    with col2:

        st.metric(
            "Input Features",
            len(FEATURES)
        )

    st.write("**Recognized classes:**")

    st.write(", ".join(map(str, MODEL_CLASSES)))

    st.caption(
        "Prediction confidence is the maximum LightGBM class "
        "probability. It is not a guarantee of correctness. "
        "Isolation Forest anomaly scores are not probabilities."
    )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8;">
        <p>🛡️ <b>CyberShield AI</b></p>
        <p>AI-Based Cyberattack Detection and Anomaly Monitoring</p>
        <p>Built using Python, LightGBM, Isolation Forest and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)



###python -m streamlit run .\app\app_v2.py --server.port 8502