import pandas as pd


def predict_one(bundle, values):
    features = bundle["features"]
    row = pd.DataFrame([{key: values[key] for key in features}], columns=features)
    encoded = pd.get_dummies(row, drop_first=True)
    encoded = encoded.reindex(columns=bundle["encoded_columns"], fill_value=0)
    model = bundle["model"]
    label = int(model.predict(encoded)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(model.predict_proba(encoded)[0][classes.index(1)])
    return label, probability


def predict_dataframe(bundle, df):
    features = bundle["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError("Missing model features: " + ", ".join(missing))
    encoded = pd.get_dummies(df[features], drop_first=True)
    encoded = encoded.reindex(columns=bundle["encoded_columns"], fill_value=0)
    result = df.copy()
    result["Predicted_SLA_Breached"] = bundle["model"].predict(encoded).astype(int)
    if hasattr(bundle["model"], "predict_proba"):
        classes = list(bundle["model"].classes_)
        if 1 in classes:
            result["Predicted_Breach_Probability"] = bundle["model"].predict_proba(encoded)[:, classes.index(1)]
    return result
