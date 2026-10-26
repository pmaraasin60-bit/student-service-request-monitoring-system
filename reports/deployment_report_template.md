# Short Deployment Report (complete after publishing)

**Project:** Student Service Request Monitoring System  
**Student:** Precious Del I. Maraasin · BSIS-3B

## Code reused
The original `Midterm_Examination.ipynb` synthetic data generation, cleaning, feature engineering, `SLA_Breached` target definition, seven submission-time predictors, 80/20 stratified split, and depth-4 Decision Tree approach were used as the basis. A fitted model bundle is saved in `models/trained_model.joblib` so Streamlit does not retrain on each refresh.

## Changes made
Added six navigation views, CSV validation and preview, interactive EDA, single and batch inference, dashboard/review queue, SAD requirements and workflow, README, requirements, and deployment instructions.

## Reproduced notebook split metrics
- Accuracy: 0.6212
- Precision (class 1): 0.6667
- Recall (class 1): 0.2759
- F1-score (class 1): 0.3902
- Train rows: 262 · Test rows: 66

## Tests performed locally
Model bundle load, notebook dataset regeneration, feature-column alignment, and generation of five sample predictions. Before submission, additionally test all Streamlit pages, invalid/missing-column CSVs, empty files, and deployment logs.

## Limitations and reflection
The source dataset is synthetic. Predictions are decision-support only and require authorized staff review. The supplied PDF report's 500-row metrics conflict with the actual notebook's 400-row data generation and seven predictors; reconcile these source materials before claiming those PDF metrics belong to this deployed model.

## Publishing fields
- GitHub repository URL: **Add after creating repository**
- Streamlit app URL: **Add after successful deployment**
