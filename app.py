import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Diabetic Readmission Risk", layout="wide")

# --- Custom styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
    }

    .main {
        background-color: #EDEFF2;
    }

    [data-testid="stMetric"] {
        background-color: #2E3440;
        border: 1px solid #3B4252;
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }

    [data-testid="stMetricValue"] {
        color: #88C0D0;
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] {
        color: #D8DEE9;
    }

    h1 {
        color: #2E3440;
        font-weight: 700;
    }

    h2, h3 {
        color: #4C566A;
        font-weight: 600;
    }

    .stAlert {
        border-radius: 14px;
        font-family: 'Quicksand', sans-serif;
    }
    </style>
""", unsafe_allow_html=True) 

df = pd.read_csv('dashboard_data.csv')
model = joblib.load('xgb_model.pkl')

st.title("🏥 Diabetic Patient Readmission Risk")
st.caption("Who's at risk of returning within 30 days, and why")

col1, col2, col3 = st.columns(3)
col1.metric("Patients", f"{len(df):,}")
col2.metric("Readmitted <30d", f"{df['readmit_30d'].mean()*100:.1f}%")
col3.metric("Model ROC-AUC", "0.66")

st.subheader("📈 Readmission rate by prior inpatient visits")
rate_by_visits = df.groupby('number_inpatient')['readmit_30d'].mean().reset_index()
fig = px.bar(
    rate_by_visits, x='number_inpatient', y='readmit_30d',
    labels={'readmit_30d': 'Readmission rate', 'number_inpatient': 'Prior inpatient visits'},
    color_discrete_sequence=['#5E81AC']
)
fig.update_layout(
    plot_bgcolor='white', paper_bgcolor='white',
    font=dict(family='Quicksand', color='#2E3440', size=14),
    xaxis=dict(title_font=dict(color='#2E3440'), tickfont=dict(color='#2E3440')),
    yaxis=dict(title_font=dict(color='#2E3440'), tickfont=dict(color='#2E3440')),
    bargap=0.25
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top risk drivers")
importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
top10 = importances.sort_values(ascending=False).head(10).reset_index()
top10.columns = ['Feature', 'Importance']
fig2 = px.bar(
    top10, x='Importance', y='Feature', orientation='h',
    color='Importance', color_continuous_scale=['#A3BE8C', '#5E81AC']
)
fig2.update_layout(
    plot_bgcolor='white', paper_bgcolor='white',
    font=dict(family='Quicksand', color='#2E3440', size=14),
    xaxis=dict(title_font=dict(color='#2E3440'), tickfont=dict(color='#2E3440')),
    yaxis=dict(title_font=dict(color='#2E3440'), tickfont=dict(color='#2E3440'), categoryorder='total ascending'),
)
st.plotly_chart(fig2, use_container_width=True)

st.info("**Takeaway:** Patients with multiple prior inpatient visits are by far the clearest group to prioritize for follow-up care.")