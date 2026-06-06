import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🔒 Fraud Detection", layout="wide")
st.title("🔒 Fraud Detection System")
st.markdown("Real-time transaction fraud detection using ML")

@st.cache_data
def generate_transaction_data(n=5000):
    np.random.seed(42)
    normal = pd.DataFrame({
        "amount": np.random.lognormal(4, 1, int(n*0.97)),
        "hour": np.random.randint(0, 24, int(n*0.97)),
        "merchant_category": np.random.randint(0, 10, int(n*0.97)),
        "distance_from_home": np.random.exponential(30, int(n*0.97)),
        "num_transactions_today": np.random.randint(1, 10, int(n*0.97)),
        "fraud": 0
    })
    fraud = pd.DataFrame({
        "amount": np.random.lognormal(7, 2, int(n*0.03)),
        "hour": np.random.choice([1,2,3,4,23], int(n*0.03)),
        "merchant_category": np.random.randint(0, 10, int(n*0.03)),
        "distance_from_home": np.random.exponential(500, int(n*0.03)),
        "num_transactions_today": np.random.randint(5, 30, int(n*0.03)),
        "fraud": 1
    })
    return pd.concat([normal, fraud]).sample(frac=1).reset_index(drop=True)

df = generate_transaction_data()
tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "🤖 Detection Model", "🔮 Real-time Check"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Fraudulent", df["fraud"].sum())
    col3.metric("Fraud Rate", f"{df['fraud'].mean():.2%}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[df.fraud==0]["amount"], bins=50, alpha=0.7, label="Normal", color="green")
    axes[0].hist(df[df.fraud==1]["amount"], bins=50, alpha=0.7, label="Fraud", color="red")
    axes[0].set_title("Transaction Amount Distribution"); axes[0].legend(); axes[0].set_xscale("log")
    df.groupby("hour")["fraud"].mean().plot(ax=axes[1], kind="bar", color="orange")
    axes[1].set_title("Fraud Rate by Hour")
    st.pyplot(fig)

with tab2:
    method = st.selectbox("Detection Method:", ["Random Forest (Supervised)", "Isolation Forest (Unsupervised)"])
    if st.button("🚀 Train & Evaluate"):
        X = df[["amount","hour","merchant_category","distance_from_home","num_transactions_today"]]
        y = df["fraud"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if "Supervised" in method:
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
            model.fit(X_tr, y_tr); y_pred = model.predict(X_te)
            st.text(classification_report(y_te, y_pred, target_names=["Normal","Fraud"]))
            st.session_state["fraud_model"] = model; st.session_state["scaler"] = scaler
        else:
            model = IsolationForest(contamination=0.03, random_state=42)
            preds = model.fit_predict(X_scaled)
            flagged = (preds == -1).sum()
            st.success(f"Isolation Forest flagged {flagged} anomalies ({flagged/len(df):.2%})")

with tab3:
    st.subheader("Check a Transaction")
    c1, c2 = st.columns(2)
    amt = c1.number_input("Amount ($)", 1.0, 50000.0, 150.0)
    hour = c2.slider("Hour of day", 0, 23, 14)
    dist = c1.number_input("Distance from home (km)", 0.0, 5000.0, 10.0)
    n_tx = c2.slider("Transactions today", 1, 30, 3)
    
    if st.button("🔍 Check for Fraud") and "fraud_model" in st.session_state:
        inp = st.session_state["scaler"].transform([[amt, hour, 3, dist, n_tx]])
        prob = st.session_state["fraud_model"].predict_proba(inp)[0][1]
        if prob > 0.7: st.error(f"🚨 HIGH FRAUD RISK: {prob:.1%} — Transaction blocked!")
        elif prob > 0.3: st.warning(f"⚠️ MEDIUM RISK: {prob:.1%} — Requires review")
        else: st.success(f"✅ LOW RISK: {prob:.1%} — Transaction approved")
