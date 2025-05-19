import streamlit as st
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

st.set_page_config(page_title="Data Analysis App", layout="wide")

st.title("🔬 Multi-Modal Bioinformatics App")

# File uploader (visible on all tabs)
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_csv = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    uploaded_h5ad = st.file_uploader("Upload .h5ad (AnnData)", type=["h5ad"])

# Load data
df = None
adata = None

if uploaded_csv is not None:
    if uploaded_csv.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_csv)
    else:
        df = pd.read_csv(uploaded_csv)
    st.sidebar.success("📊 Table uploaded!")

if uploaded_h5ad is not None:
    adata = sc.read(uploaded_h5ad)
    st.sidebar.success("🧬 AnnData uploaded!")

# Tabs
tabs = st.tabs(["🏠 Home", "🧹 Preprocessing", "🤖 ML Pipeline", "👥 Team Information"])

# ---------------- Tab 1: Home ----------------
with tabs[0]:
    st.header("Welcome")
    st.markdown("""
        This application supports CSV, Excel, and AnnData (.h5ad) formats.

        - **Preprocessing**: Clean and explore your dataset.
        - **ML Pipeline**: Run basic classification or regression models.
        - **Differential Expression (DEG)**: For single-cell datasets with AnnData.
    """)

# ---------------- Tab 2: Preprocessing ----------------
with tabs[1]:
    st.header("🧹 Preprocessing")

    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=False)

        st.subheader("Basic statistics")
        st.dataframe(df.describe(), use_container_width=False)

        st.subheader("Missing values")
        st.dataframe(df.isnull().sum().to_frame("Missing Count"), use_container_width=False)

    elif adata is not None:
        st.subheader("Scanpy preprocessing steps")
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        st.success("Normalization, log1p, and HVG extraction complete.")
        st.text(adata)

# ---------------- Tab 3: ML Pipeline ----------------
with tabs[2]:
    st.header("🤖 Machine Learning Pipeline")

    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Column", df.columns)
        with col2:
            test_size = st.slider("Test Split %", 0.1, 0.5, 0.3)

        features = [col for col in df.columns if col != target]
        X = pd.get_dummies(df[features])
        y = df[target]

        if y.dtype == "object" or y.nunique() < 15:
            task_type = "classification"
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            task_type = "regression"

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        if task_type == "classification":
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"🔍 Classification Accuracy: {acc:.2f}")

            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        else:
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            st.success(f"📈 Regression R² Score: {r2:.2f}")
            st.info(f"Mean Squared Error: {mse:.2f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, preds, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Predicted vs Actual")
            st.pyplot(fig)


# ---------------- Tab 4 ----------------

with tabs[3]:
    st.header("👥 Team Information")
    st.markdown("""
    - **George** – ML Developer  
    """)

