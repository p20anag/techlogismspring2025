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

tabs = st.tabs([
    "📁 Upload & Preview",
    "🧹 Preprocessing",
    "🤖 ML Pipeline",
    "🧬 DEG Analysis",
    "📊 Expression Plots",
    "🐳 Docker Info",
    "👥 Team"
])

uploaded_csv = None
uploaded_h5ad = None
adata = st.session_state.get("adata", None)
df = None

# Tab 1 - Upload
with tabs[0]:
    st.header("📁 Upload & Preview")
    tab_choice = st.radio("Choose data type", ["Tabular Data", "AnnData (.h5ad)"])
    if tab_choice == "Tabular Data":
        uploaded_tabular = st.file_uploader("Upload CSV / TXT / Excel file", type=["csv", "txt", "xlsx"])
        if uploaded_tabular:
            file_name = uploaded_tabular.name
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_tabular)
            elif file_name.endswith(".txt"):
                df = pd.read_csv(uploaded_tabular, sep="\t")
            elif file_name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_tabular)
            else:
                st.error("Unsupported format.")
            if df is not None:
                st.success("✅ File uploaded")
                st.dataframe(df.head(), height=250)
    elif tab_choice == "AnnData (.h5ad)":
        uploaded_h5ad = st.file_uploader("Upload .h5ad", type=["h5ad"])
        if uploaded_h5ad:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad") as tmp_file:
                tmp_file.write(uploaded_h5ad.read())
                tmp_file_path = tmp_file.name
            adata = sc.read(tmp_file_path)
            st.session_state["adata"] = adata
            st.success(f"✅ Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
            st.text(adata)

# Tab 2 - Preprocessing
with tabs[1]:
    st.header("🧹 Preprocessing")
    adata = st.session_state.get("adata", None)
    if adata is not None:
        min_genes = st.slider("Min Genes per Cell", 200, 1000, 600)
        min_cells = st.slider("Min Cells per Gene", 1, 10, 3)
        if st.button("Run Preprocessing"):
            original_obs = adata.obs.copy()
            sc.pp.filter_cells(adata, min_genes=min_genes)
            sc.pp.filter_genes(adata, min_cells=min_cells)
            adata = adata[:, [g for g in adata.var_names if not str(g).startswith(('ERCC', 'MT-', 'mt-'))]]
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            for col in original_obs.columns:
                if col not in adata.obs.columns:
                    adata.obs[col] = original_obs[col].reindex(adata.obs.index)
            adata.raw = adata.copy()
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata = adata[:, adata.var.highly_variable].copy()
            sc.pp.scale(adata, max_value=10)
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            st.session_state["adata"] = adata
            st.success("✅ Preprocessing Complete.")
    else:
        st.warning("⚠️ Upload a .h5ad file first.")

# Tab 3 - ML
with tabs[2]:
    st.header("🤖 Machine Learning Pipeline")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Column", df.columns)
        with col2:
            test_size = st.slider("Test Split %", 0.1, 0.5, 0.3)
        features = [c for c in df.columns if c != target]
        X = pd.get_dummies(df[features])
        y = df[target]
        task_type = "classification" if y.dtype == "object" or y.nunique() < 15 else "regression"
        if task_type == "classification":
            y = LabelEncoder().fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        model = RandomForestClassifier() if task_type == "classification" else RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if task_type == "classification":
            acc = accuracy_score(y_test, preds)
            st.success(f"🔍 Accuracy: {acc:.2f}")
            sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues")
        else:
            st.success(f"📈 R²: {r2_score(y_test, preds):.2f}")
            st.info(f"MSE: {mean_squared_error(y_test, preds):.2f}")
        st.pyplot()


# Tab 4 - DEG
with tabs[3]:
    st.header("🧬 Differential Expression (DEG)")
    adata = st.session_state.get("adata", None)
    if adata is not None and adata.obs is not None:
        cluster_col = st.selectbox("Group by", adata.obs.columns)
        group_vals = adata.obs[cluster_col].unique().tolist()
        if len(group_vals) >= 2:
            group = st.selectbox("Test group", group_vals)
            ref = st.selectbox("Reference", [g for g in group_vals if g != group])
            method = st.radio("Method", ["t-test", "wilcoxon"])
            group_size = (adata.obs[cluster_col] == group).sum()
            ref_size = (adata.obs[cluster_col] == ref).sum()
            st.write(f"Group: {group_size}, Reference: {ref_size}")

            if group_size < 3 or ref_size < 3:
                st.error("⚠️ Too few cells in group/reference. Minimum is 3.")
            elif st.button("Run DEG Analysis"):
                try:
                    current_adata = adata.raw.to_adata() if adata.raw is not None else adata.copy()
                    current_adata.obs[cluster_col] = adata.obs[cluster_col]
                    sc.tl.rank_genes_groups(
                        current_adata,
                        groupby=cluster_col,
                        method=method,
                        groups=[group],
                        reference=ref,
                        use_raw=False
                    )
                    adata.uns["rank_genes_groups"] = current_adata.uns["rank_genes_groups"]
                    st.session_state["adata"] = adata
                    st.session_state["deg_group"] = group

                    result_df = sc.get.rank_genes_groups_df(current_adata, group=group)
                    st.success("✅ DEG Complete")
                    st.dataframe(result_df.head())
                except Exception as e:
                    st.error(f"❌ DEG failed: {e}")

            if st.button("Generate Volcano Plot"):
                try:
                    group = st.session_state.get("deg_group", group)
                    if "rank_genes_groups" not in adata.uns:
                        st.error("❌ No DEG results found. Please run DEG first.")
                    else:
                        result_df = sc.get.rank_genes_groups_df(adata, group=group).dropna()

                        result_df["neg_log10_pval"] = -np.log10(result_df["pvals"])
                        result_df["diffexpressed"] = "NS"
                        result_df.loc[(result_df["logfoldchanges"] > 1) & (result_df["pvals_adj"] < 0.05), "diffexpressed"] = "UP"
                        result_df.loc[(result_df["logfoldchanges"] < -1) & (result_df["pvals_adj"] < 0.05), "diffexpressed"] = "DOWN"

                        fig, ax = plt.subplots(figsize=(12, 5))
                        sns.scatterplot(
                            data=result_df,
                            x="logfoldchanges",
                            y="neg_log10_pval",
                            hue="diffexpressed",
                            palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "gray"},
                            alpha=0.7,
                            ax=ax
                        )
                        ax.axhline(-np.log10(0.05), color='gray', linestyle='--')
                        ax.axvline(-1, color='gray', linestyle='--')
                        ax.axvline(1, color='gray', linestyle='--')
                        ax.set_xlim(-11, 11)
                        ax.set_title(f"Volcano of DEGs – {group}")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ Volcano failed: {e}")
    else:
        st.warning("⚠️ Upload a .h5ad file first.")

# Tab 5 - Gene Expression Plot
with tabs[4]:
    st.header("📊 Gene Expression Visualization")
    adata = st.session_state.get("adata", None)
    if adata is not None:
        gene = st.text_input("Enter gene name")
        if gene:
            try:
                if gene not in adata.var_names:
                    st.error(f"Gene {gene} not found in dataset")
                else:
                    if "neighbors" not in adata.uns:
                        sc.pp.neighbors(adata)
                    if "X_umap" not in adata.obsm:
                        sc.tl.umap(adata)

                    col1, col2 = st.columns(2)

                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(8, 4))
                        sc.pl.violin(adata, gene, groupby=adata.obs.columns[0], show=False, ax=ax1)
                        st.pyplot(fig1)

                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        sc.pl.umap(adata, color=gene, show=False, ax=ax2)
                        st.pyplot(fig2)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Tab 6 - Docker
with tabs[5]:
    st.header("🐳 Dockerization Overview")
    st.markdown("""
    ```Dockerfile
    FROM python:3.10
    WORKDIR /app
    COPY . /app
    RUN pip install -r requirements.txt
    CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
    ```
    Run it:
    ```bash
    docker build -t bio-app .
    docker run -p 8501:8501 bio-app
    ```
    """)

# Tab 7 - Team
with tabs[6]:
    st.header("👥 Team")
    st.markdown("""
    - **George** – ML Developer
    """)

