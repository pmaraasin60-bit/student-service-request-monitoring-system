import pandas as pd

REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days", "Status",
    "Reopened", "Satisfaction_Score"
]
FEATURES = ["Service_Type", "Requester_Type", "Requested_Channel", "Priority",
            "Queue_Length", "Required_Documents_Complete", "SLA_Days"]


def validate_dataset(df):
    if df is None or df.empty:
        return False, ["The uploaded CSV is empty." ]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, ["Missing required columns: " + ", ".join(missing)]
    return True, []


def prepare_dataframe(df):
    """Apply the midterm notebook's basic cleaning and derived fields."""
    data = df.copy()
    data = data.drop_duplicates(subset=["Request_ID"], keep="first")
    text_cols = ["Service_Type", "Requester_Type", "Requested_Channel", "Priority",
                 "Assigned_Office", "Required_Documents_Complete", "Status", "Reopened"]
    for col in text_cols:
        data[col] = data[col].astype("string").str.title().str.strip()
    data["Submission_Date"] = pd.to_datetime(data["Submission_Date"], errors="coerce")
    for col in ["Queue_Length", "SLA_Days", "Processing_Days"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data[data["Submission_Date"].notna()]
    data = data[(data["Queue_Length"] >= 0) & (data["SLA_Days"] >= 0)]
    data["Submission_Weekday"] = data["Submission_Date"].dt.day_name()
    data["Queue_Category"] = pd.cut(data["Queue_Length"], bins=[-1, 5, 15, float("inf")],
                                    labels=["Low", "Medium", "High"])
    completed = (data["Status"] == "Completed") & data["Processing_Days"].notna() & data["SLA_Days"].notna()
    data["SLA_Breached"] = pd.NA
    data.loc[completed, "SLA_Breached"] = (data.loc[completed, "Processing_Days"] > data.loc[completed, "SLA_Days"]).astype(int)
    return data
