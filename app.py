from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from modules.data_processing import prepare_dataframe, validate_dataset, FEATURES
from modules.exploratory_analysis import class_distribution, queue_breach_chart, office_chart, service_volume_chart
from modules.prediction import predict_one, predict_dataframe
from modules.dashboard import make_dashboard, risk_queue

ROOT = Path(__file__).parent
st.set_page_config(page_title="Student Service Request Monitoring System", page_icon="📊", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "trained_model.joblib")

@st.cache_data
def load_sample_data():
    return pd.read_csv(ROOT / "data" / "student_service_requests.csv")

bundle = load_bundle()
st.sidebar.title("Service Request Monitor")
st.sidebar.caption("Section B · SLA breach decision support")
page = st.sidebar.selectbox("Navigate", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
if "working_df" not in st.session_state:
    st.session_state.working_df = prepare_dataframe(load_sample_data())
if "upload_name" not in st.session_state:
    st.session_state.upload_name = "Included synthetic midterm dataset"
df = st.session_state.working_df

if page == "Home":
    st.title("Student Service Request Monitoring System")
    st.write("A Streamlit decision-support application based on the uploaded midterm notebook. It estimates whether a completed service request may breach its SLA using submission-time information.")
    a,b,c = st.columns(3)
    a.metric("Dataset rows", f"{len(df):,}")
    b.metric("Labeled completed requests", f"{int(df['SLA_Breached'].notna().sum()):,}")
    c.metric("Model", "Decision Tree · depth 4")
    st.info("Use predictions to prioritize staff review. A risk flag is not a denial or a final administrative decision.")
    st.subheader("Available modules")
    st.markdown("""- **Dataset:** upload/validate CSV, preview records, inspect missing values and duplicate IDs.
- **Data Analysis:** descriptive statistics and three interactive charts.
- **Prediction:** enter one request or score a batch CSV.
- **Dashboard:** view workload, high-queue items, and a review queue.
- **System Design:** functional/non-functional requirements, actors, use case diagram, and decision workflow.""")
    st.caption(f"Model file: models/trained_model.joblib · Loaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

elif page == "Dataset":
    st.title("Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload a service-request CSV", type=["csv"])
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            valid, issues = validate_dataset(incoming)
            if valid:
                st.session_state.working_df = prepare_dataframe(incoming)
                st.session_state.upload_name = uploaded.name
                df = st.session_state.working_df
                st.success(f"Loaded and cleaned {uploaded.name}.")
            else:
                for issue in issues: st.error(issue)
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    st.caption(f"Current data source: {st.session_state.upload_name}")
    st.subheader("Data preview")
    st.dataframe(df.head(25), use_container_width=True)
    x,y,z = st.columns(3)
    x.metric("Rows after cleaning", len(df))
    y.metric("Duplicate Request_IDs remaining", int(df["Request_ID"].duplicated().sum()))
    z.metric("Missing cells", int(df.isna().sum().sum()))
    st.subheader("Missing values by column")
    st.dataframe(df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
    st.download_button("Download cleaned data", df.to_csv(index=False).encode("utf-8"), "cleaned_service_requests.csv", "text/csv")
    if st.button("Restore included synthetic dataset"):
        st.session_state.working_df = prepare_dataframe(load_sample_data())
        st.session_state.upload_name = "Included synthetic midterm dataset"
        st.rerun()

elif page == "Data Analysis":
    st.title("Exploratory Data Analysis")
    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    st.subheader("Target class distribution")
    dist = class_distribution(df)
    if not dist.empty:
        dist["Class"] = dist["SLA_Breached"].map({0:"No SLA breach",1:"SLA breached"})
        st.plotly_chart(px.pie(dist, names="Class", values="Count", title="Labeled completed requests"), use_container_width=True)
    for chart_fn in [queue_breach_chart, office_chart, service_volume_chart]:
        chart = chart_fn(df)
        if chart is not None: st.plotly_chart(chart, use_container_width=True)
    st.caption("The target is available only for completed requests with valid Processing_Days and SLA_Days. These post-completion fields are excluded from model inputs.")

elif page == "Prediction":
    st.title("Predict SLA Breach Risk")
    st.write("Enter information available when the request is submitted. Predictions are based on the saved Decision Tree trained from the midterm notebook's synthetic dataset.")
    with st.form("single_prediction"):
        col1,col2,col3 = st.columns(3)
        with col1:
            service = st.selectbox("Service type", ["Records", "Certification", "IT Support", "Scholarship Inquiry", "Library Access"])
            requester = st.selectbox("Requester type", ["Student", "Faculty", "Staff"])
            channel = st.selectbox("Requested channel", ["Portal", "Email", "Walk-in"])
        with col2:
            priority = st.selectbox("Priority", ["Low", "Normal", "High"])
            queue = st.number_input("Queue length at submission", min_value=0, max_value=10000, value=10, step=1)
        with col3:
            docs = st.selectbox("Required documents complete?", ["Yes", "No"])
            sla = st.selectbox("SLA days", [2,3,5,7], index=1)
        submitted = st.form_submit_button("Predict risk")
    if submitted:
        values = {"Service_Type":service,"Requester_Type":requester,"Requested_Channel":channel,"Priority":priority,"Queue_Length":int(queue),"Required_Documents_Complete":docs,"SLA_Days":int(sla)}
        label, probability = predict_one(bundle, values)
        st.subheader("Prediction result")
        if label == 1: st.warning("Predicted class: SLA breach risk")
        else: st.success("Predicted class: No SLA breach")
        if probability is not None: st.metric("Estimated breach probability", f"{probability:.1%}")
        factors=[]
        if queue > 15: factors.append("queue length is above 15")
        if docs == "No": factors.append("required documents are incomplete")
        if factors: st.write("Review prompts: " + "; ".join(factors) + ".")
        st.caption("This is a model estimate, not a guarantee. Authorized staff must verify the request and decide follow-up actions.")
    st.divider()
    st.subheader("Batch predictions")
    st.write("Upload a CSV containing the seven submission-time model features. Extra columns are retained, but only the seven model features are used.")
    batch = st.file_uploader("Upload prediction inputs", type=["csv"], key="batch_upload")
    if batch is not None:
        try:
            batch_df = pd.read_csv(batch)
            missing = [col for col in FEATURES if col not in batch_df.columns]
            if batch_df.empty: st.error("The prediction CSV is empty.")
            elif missing: st.error("Missing model feature columns: " + ", ".join(missing))
            else:
                result = predict_dataframe(bundle, batch_df)
                st.dataframe(result, use_container_width=True)
                st.download_button("Download predictions", result.to_csv(index=False).encode("utf-8"), "sla_breach_predictions.csv", "text/csv")
        except Exception as exc: st.error(f"Batch prediction failed: {exc}")
    st.subheader("Five sample predictions")
    sample = df[FEATURES].head(5).copy()
    sample_result = predict_dataframe(bundle, sample)
    st.dataframe(sample_result, use_container_width=True)

elif page == "Dashboard":
    st.title("Decision-Support Dashboard")
    stats = make_dashboard(df)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Requests", f"{stats['total']:,}")
    c2.metric("Pending requests", f"{stats['pending']:,}")
    c3.metric("Historical SLA breach rate", f"{stats['breach_rate']:.1%}")
    c4.metric("Queue length above 15", f"{stats['high_queue']:,}")
    st.subheader("Operational review queue")
    review = df.copy()
    review["Review_Reason"] = review.apply(lambda r: "High queue" if pd.notna(r.get("Queue_Length")) and r["Queue_Length"] > 15 else ("Documents incomplete" if r.get("Required_Documents_Complete") == "No" else "Routine review"), axis=1)
    review = review[(review["Status"] == "Pending") | (review["Queue_Length"] > 15) | (review["Required_Documents_Complete"] == "No")]
    st.dataframe(review[["Request_ID","Service_Type","Assigned_Office","Status","Queue_Length","SLA_Days","Required_Documents_Complete","Review_Reason"]].head(50), use_container_width=True)
    st.markdown("**Suggested human follow-up:** verify missing documents, confirm the current office queue, and consider escalation or workload reassignment for urgent requests. Risk indicators should never automatically reject a request.")
    st.subheader("Predicted risk on current rows")
    scored = predict_dataframe(bundle, df[FEATURES].copy())
    st.dataframe(scored.head(25), use_container_width=True)
    st.download_button("Download scored rows", scored.to_csv(index=False).encode("utf-8"), "scored_service_requests.csv", "text/csv")

elif page == "System Design":
    st.title("Systems Analysis and Design Integration")
    st.subheader("Functional requirements")
    requirements = [
        ("FR-01", "Validate uploaded CSV schema and show missing/duplicate data summaries."),
        ("FR-02", "Estimate SLA_Breached using the saved Decision Tree and submission-time features only."),
        ("FR-03", "Show descriptive statistics, class distribution, and at least three charts."),
        ("FR-04", "Display workload KPIs and a staff review queue for high-queue or incomplete-document requests."),
        ("FR-05", "Present use cases and recommended human follow-up; never auto-deny service requests."),
    ]
    st.table(pd.DataFrame(requirements, columns=["ID","Requirement"]))
    st.subheader("Non-functional requirements")
    st.markdown("""- **Reliability:** handle empty/invalid CSV files and report missing columns.
- **Maintainability:** separate data processing, analysis, prediction, and dashboard logic into modules.
- **Privacy and governance:** deploy only synthetic or authorized data; keep final decisions with authorized staff.""")
    st.subheader("Actors")
    st.markdown("""- **Student / Requester:** submits a request and tracks status in the operational system.
- **Office Staff:** processes requests and reviews delay-risk indicators.
- **System Administrator:** reviews SLA analytics and oversees model/system configuration.""")
    st.subheader("Use case diagram")
    image_path = ROOT / "assets" / "use_case_diagram.png"
    if image_path.exists(): st.image(str(image_path), caption="Provided UML use case diagram", use_container_width=True)
    st.subheader("Decision-support workflow")
    st.markdown("""1. Receive a request and capture submission-time fields.
2. Validate required data and document completeness.
3. Encode the seven model features using the saved training columns.
4. Predict SLA breach risk and display the result with context.
5. Office staff reviews the flag, verifies current conditions, and records a human decision.
6. Monitor queue and SLA performance; use outcomes to improve service processes.""")
    st.subheader("Model details and limitations")
    st.json({"target": bundle["target"], "features": bundle["features"], "model": bundle["model_version"], "metrics_from_recreated_midterm_split": bundle["metrics"]})
    st.warning("The uploaded notebook generates 400 synthetic records and trains using seven predictors. The accompanying PDF report describes a different 500-record analysis and a different feature set/metric table. This application follows the actual uploaded notebook code as required; reconcile the report before final submission.")
