import streamlit as st
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy.external as sce
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, r2_score

st.set_page_config(page_title="BioData Pipeline", layout="wide")

# --- Tabs ---
tabs = st.tabs([
    "📁 Upload & Preview",
    "🧹 Preprocessing",
    "🤖 ML Pipeline",
    "🧬 DEG Analysis",
    "📊 Expression Plots",
    "🐳 Docker Info",
    "👥 Team"
])

# Shared variables
uploaded_csv = None
uploaded_h5ad = None
adata = None
df = None

# ---------------- Tab 1 ----------------
tab1, tab2 = st.tabs(["📄 Tabular Data", "🧫 AnnData (.h5ad)"])

with tab1:
    uploaded_tabular = st.file_uploader("Upload CSV / TXT / Excel file", type=["csv", "txt", "xlsx"])
    if uploaded_tabular is not None:
        file_name = uploaded_tabular.name
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_tabular)
        elif file_name.endswith(".txt"):
            try:
                df = pd.read_csv(uploaded_tabular, sep="\t")
            except:
                df = pd.read_csv(uploaded_tabular, delimiter=None, engine="python")
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_tabular)
        else:
            st.error("Unsupported tabular format.")
            df = None
        if df is not None:
            st.success("✅ Tabular file uploaded successfully!")
            st.dataframe(df.head(), height=250)

with tab2:
    uploaded_h5ad = st.file_uploader("Upload .h5ad (AnnData)", type=["h5ad"])
    if uploaded_h5ad is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad") as tmp_file:
            tmp_file.write(uploaded_h5ad.read())
            tmp_file_path = tmp_file.name
        adata = sc.read(tmp_file_path)
        st.success(f"✅ AnnData loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
        st.write("AnnData summary:")
        st.text(adata)

if adata is not None:
    st.subheader("🔬 First 20 Gene Names")
    st.write(adata.var_names[:20].tolist())

# ---------------- Tab 2 ----------------
with tabs[1]:
    st.header("🧹 Preprocessing")
    if adata is not None:
        min_genes = st.slider("Min Genes per Cell", 200, 1000, 600)
        min_cells = st.slider("Min Cells per Gene", 1, 10, 3)
        if st.button("Run Preprocessing"):
            sc.pp.filter_cells(adata, min_genes=min_genes)
            sc.pp.filter_genes(adata, min_cells=min_cells)
            adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(('ERCC', 'MT-', 'mt-'))]]
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata.raw = adata
            adata = adata[:, adata.var.highly_variable]
            sc.pp.scale(adata, max_value=10)
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            st.success("✅ Preprocessing Complete.")
        if st.button("Integrate Batches with Harmony"):
            if "X_pca" not in adata.obsm:
                st.warning("PCA not found. Running PCA first...")
                sc.pp.pca(adata)
            sce.pp.harmony_integrate(adata, key='batch')
            sc.pp.neighbors(adata, use_rep='X_pca_harmony')
            sc.tl.umap(adata)
            st.success("✅ Harmony Integration Complete.")
            fig, ax = plt.subplots()
            sc.pl.umap(adata, color=["batch"], show=False, ax=ax)
            st.pyplot(fig)

# ---------------- Tab 3 ----------------
with tabs[2]:
    st.header("🤖 Machine Learning Pipeline")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Column", df.columns)
        with col2:
            test_size = st.slider("Test Split %", 0.1, 0.5, 0.3)
        features = [col for col in df.columns if col != target]
        X = df[features]
        y = df[target]
        X = pd.get_dummies(X)
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
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
        else:
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            st.success(f"📈 Regression R² Score: {r2:.2f}")
            st.info(f"Mean Squared Error: {mse:.2f}")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, preds, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Predicted vs Actual")
            st.pyplot(fig)

# ---------------- Tab 4 ----------------
with tabs[3]:
    st.header("🧬 Differential Expression (DEG)")
    if adata is not None and adata.obs is not None:
        cluster_col = st.selectbox("Group by column", adata.obs.columns)
        if cluster_col:
            group_values = list(adata.obs[cluster_col].unique())
            if len(group_values) >= 2:
                group = st.selectbox("Group (test)", group_values)
                reference = st.selectbox("Reference", [val for val in group_values if val != group])
                method = st.radio("Method", ["t-test", "wilcoxon"])
                if st.button("Run DEG Analysis"):
                    try:
                        sc.tl.rank_genes_groups(adata, groupby=cluster_col, method=method,
                                                groups=[group], reference=reference, use_raw=False)
                        result_df = sc.get.rank_genes_groups_df(adata, group=None)
                        st.session_state["deg_result"] = result_df
                        st.session_state["deg_group"] = group
                        st.success("✅ DEG Analysis Completed.")
                        st.dataframe(result_df.head(), height=250)
                    except Exception as e:
                        st.error(f"DEG Analysis failed: {e}")
                if st.button("Generate Volcano Plot"):
                    result = adata.uns.get("rank_genes_groups", None)
                    group = st.session_state.get("deg_group")
                    if result and group:
                        degs_df = pd.DataFrame({
                            "genes": result["names"][group],
                            "pvals": result["pvals"][group],
                            "pvals_adj": result["pvals_adj"][group],
                            "logfoldchanges": result["logfoldchanges"][group],
                        })
                        degs_df["neg_log10_pval"] = -np.log10(degs_df["pvals"])
                        degs_df["diffexpressed"] = "NS"
                        degs_df.loc[(degs_df["logfoldchanges"] > 1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "UP"
                        degs_df.loc[(degs_df["logfoldchanges"] < -1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "DOWN"
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(data=degs_df, x="logfoldchanges", y="neg_log10_pval", hue="diffexpressed",
                                        palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "grey"}, alpha=0.7, ax=ax)
                        ax.axhline(-np.log10(0.05), color='gray', linestyle='dashed')
                        ax.axvline(-1, color='gray', linestyle='dashed')
                        ax.axvline(1, color='gray', linestyle='dashed')
                        ax.set_xlim(-11, 11)
                        ax.set_title("Volcano of DEGs")
                        st.pyplot(fig)
                    else:
                        st.error("Run DEG Analysis first to compute differential expression.")

# ---------------- Tab 5 ----------------
with tabs[4]:
    st.header("📊 Gene Expression Visualization")
    if adata is not None:
        gene = st.text_input("Enter gene name to visualize:", key="gene_input")
        if gene:
            try:
                if "neighbors" not in adata.uns:
                    sc.pp.neighbors(adata)
                if "X_umap" not in adata.obsm.keys():
                    sc.tl.umap(adata)
                fig1, ax1 = plt.subplots()
                sc.pl.violin(adata, gene, groupby="celltype", show=False, ax=ax1)
                st.pyplot(fig1)
                fig2, ax2 = plt.subplots()
                sc.pl.umap(adata, color=gene, show=False, ax=ax2)
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------- Tab 6 ----------------
with tabs[5]:
    st.header("🐳 Dockerization Overview")
    st.markdown("""
    **Dockerfile snippet:**
    ```Dockerfile
    FROM python:3.10
    WORKDIR /app
    COPY . /app
    RUN pip install -r requirements.txt
    CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
    ```

    **To build and run:**
    ```bash
    docker build -t bio-app .
    docker run -p 8501:8501 bio-app
    ```
    """)

# ---------------- Tab 7 ----------------
with tabs[6]:
    st.header("👥 Team Information")
    st.markdown("""
    - **George** – ML Developer
    """)
