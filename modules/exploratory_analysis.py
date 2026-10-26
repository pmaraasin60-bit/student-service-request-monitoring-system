import pandas as pd
import plotly.express as px


def class_distribution(df):
    labeled = df[df["SLA_Breached"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame(columns=["SLA_Breached", "Count"])
    return labeled["SLA_Breached"].astype(int).value_counts().sort_index().rename_axis("SLA_Breached").reset_index(name="Count")


def queue_breach_chart(df):
    labeled = df[df["SLA_Breached"].notna()].copy()
    if labeled.empty: return None
    labeled["SLA_Breached"] = labeled["SLA_Breached"].astype(int)
    summary = labeled.groupby("Queue_Category", observed=False)["SLA_Breached"].mean().reset_index()
    summary["Breach Rate (%)"] = summary["SLA_Breached"] * 100
    return px.bar(summary, x="Queue_Category", y="Breach Rate (%)", title="SLA breach rate by queue category", text_auto=".1f")


def office_chart(df):
    labeled = df[df["SLA_Breached"].notna()].copy()
    if labeled.empty: return None
    labeled["SLA_Breached"] = labeled["SLA_Breached"].astype(int)
    summary = labeled.groupby("Assigned_Office")["SLA_Breached"].mean().mul(100).reset_index(name="Breach Rate (%)")
    return px.bar(summary, x="Assigned_Office", y="Breach Rate (%)", title="SLA breach rate by assigned office", text_auto=".1f")


def service_volume_chart(df):
    summary = df["Service_Type"].value_counts().rename_axis("Service_Type").reset_index(name="Requests")
    return px.bar(summary, x="Service_Type", y="Requests", title="Request volume by service type", text_auto=True)
