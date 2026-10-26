# Student Service Request Monitoring System

A Streamlit decision-support app for the Systems Analysis and Design Section B exercise. It reuses the **actual uploaded midterm notebook**: the notebook generates 400 synthetic student service requests, cleans the data, derives `SLA_Breached` for completed requests, and trains a depth-4 `DecisionTreeClassifier`.

## Modules
- Home
- Dataset upload, schema validation, cleaning summary, and CSV download
- Data analysis with descriptive statistics and interactive charts
- Single-request and batch SLA breach prediction
- Decision-support dashboard and review queue
- SAD requirements, actors, supplied use-case diagram, workflow, and limitations

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a new GitHub repository and upload the contents of this folder (not the ZIP itself).
2. On https://share.streamlit.io, sign in with GitHub and choose **Create app**.
3. Select the repository, branch `main`, and main file path `app.py`.
4. Deploy and test the resulting app URL.

## Model specification from the notebook
- Target: `SLA_Breached` (1 when `Processing_Days > SLA_Days`, only for completed requests)
- Predictors: `Service_Type`, `Requester_Type`, `Requested_Channel`, `Priority`, `Queue_Length`, `Required_Documents_Complete`, `SLA_Days`
- Excluded from inference: post-completion values such as `Processing_Days`, `Status`, `Reopened`, and `Satisfaction_Score`
- Model: `DecisionTreeClassifier(max_depth=4, random_state=42)`
- Saved model: `models/trained_model.joblib`

## Dataset
The included CSV is synthetic and regenerated from the uploaded midterm notebook with its fixed random seed. Do not upload real student records to a public app unless appropriately authorized.

## Important source consistency note
The supplied PDF report describes a 500-row dataset, different feature columns, and reported metrics of 84.4% accuracy. The actual uploaded notebook generates 400 rows and uses seven features. This app follows the actual notebook code; its reproduced test-split metrics are saved in `reports/model_metrics.json`. Reconcile the PDF and notebook results before submitting a final report.

## Safety and limitations
Predictions support staff review only. A predicted breach is not proof that a request will be late and must never automatically deny service. The synthetic dataset is illustrative and not a substitute for validating the model on authorized operational data.
