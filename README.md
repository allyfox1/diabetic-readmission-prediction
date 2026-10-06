# Diabetic Patient Readmission Prediction

Predicting which diabetic patients are at highest risk of being readmitted within 30 days of discharge, and identifying what hospitals could act on to reduce it.

🔗 **[Live Dashboard] https://diabetic-readmission-prediction-ggdgeaexxk9fvavjsnheog.streamlit.app/**

## The Question

Which diabetic patients are at highest risk of 30-day readmission, and what should hospitals prioritize to reduce it?

## Data

101,766 inpatient encounters across 130 US hospitals (1999–2008), from the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008).

## Approach

- Cleaned and engineered features from raw EHR data (missing value handling, diagnosis code grouping, medical specialty simplification)
- Compared a logistic regression baseline against XGBoost
- Evaluated using ROC-AUC and recall, given ~11% class imbalance
- Interpreted results with feature importance analysis
- Built an interactive Streamlit dashboard to communicate findings

## Key Findings

- XGBoost outperformed the logistic regression baseline (ROC-AUC 0.66 vs. 0.64; recall 56% vs. 52% on readmitted patients)
- **Number of prior inpatient visits** is by far the strongest predictor of readmission risk
- Other key drivers: discharge disposition, number of diagnoses, number of emergency visits, and diabetes medication management at discharge

## Recommendation

Hospitals should prioritize discharge planning and follow-up scheduling for patients with 2+ prior inpatient admissions, and ensure diabetes medication regimens are clearly managed before discharge.

## Limitations

This dataset lacks social/behavioral factors (housing stability, medication adherence, caregiver support) that likely influence readmission but aren't captured in EHR data.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly