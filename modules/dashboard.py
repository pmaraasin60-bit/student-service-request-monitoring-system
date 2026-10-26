import pandas as pd
import plotly.express as px


def make_dashboard(df, predictions=None):
    data = df.copy()
    total = len(data)
    pending = int((data["Status"] == "Pending").sum()) if "Status" in data else 0
    labeled = data[data["SLA_Breached"].notna()] if "SLA_Breached" in data else pd.DataFrame()
    breach_rate = float(pd.to_numeric(labeled["SLA_Breached"], errors="coerce").mean()) if len(labeled) else 0.0
    high_queue = int((pd.to_numeric(data["Queue_Length"], errors="coerce") > 15).sum()) if "Queue_Length" in data else 0
    return {"total": total, "pending": pending, "breach_rate": breach_rate, "high_queue": high_queue}


def risk_queue(df, predictions=None):
    data = df.copy()
    if "SLA_Breached" in data:
        data = data[data["SLA_Breached"].notna()].copy()
    if predictions is not None:
        return predictions
    return data.sort_values("Queue_Length", ascending=False).head(25)
