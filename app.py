import streamlit as st
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, r2_score

st.set_page_config(page_title="BioData Pipeline", layout="wide")

tabs = st.tabs([
    "📁 Upload & Preview",
    "🧹 Preprocessing",
    "🤖 ML Pipeline",
    "🧬 DEG + Volcano",
    "📊 Expression Plots",
    "👥 Team"
])

adata = st.session_state.get("adata", None)
df = st.session_state.get("df", None)

# Tab 1 - Upload
with tabs[0]:
    st.header("📁 Upload & Preview")
    tab_choice = st.radio("Choose data type", ["Tabular Data", "AnnData (.h5ad)"])
    if tab_choice == "Tabular Data":
        uploaded_tabular = st.file_uploader("Upload CSV / TXT / Excel file", type=["csv", "txt", "xlsx"])
        if uploaded_tabular:
            ext = uploaded_tabular.name.split(".")[-1]
            if ext == "csv":
                df = pd.read_csv(uploaded_tabular)
            elif ext == "txt":
                df = pd.read_csv(uploaded_tabular, sep="\t")
            elif ext == "xlsx":
                df = pd.read_excel(uploaded_tabular)
            else:
                st.error("Unsupported format")
            if df is not None:
                st.success("✅ File uploaded")
                st.dataframe(df.head())
                st.session_state["df"] = df
    else:
        uploaded_h5ad = st.file_uploader("Upload .h5ad", type=["h5ad"])
        if uploaded_h5ad:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad") as tmp_file:
                tmp_file.write(uploaded_h5ad.read())
                tmp_file_path = tmp_file.name
            adata = sc.read(tmp_file_path)
            st.session_state["adata"] = adata
            st.success(f"✅ Loaded {adata.n_obs} cells × {adata.n_vars} genes")
            st.dataframe(adata.obs.head())

# Tab 2 - Preprocessing
with tabs[1]:
    st.header("🧹 Preprocessing")
    adata = st.session_state.get("adata", None)
    if adata is not None:
        min_genes = st.slider("Min genes per cell", 200, 1000, 500)
        min_cells = st.slider("Min cells per gene", 1, 20, 3)
        if st.button("Run Preprocessing"):
            sc.pp.filter_cells(adata, min_genes=min_genes)
            sc.pp.filter_genes(adata, min_cells=min_cells)
            adata = adata[:, [g for g in adata.var_names if not g.startswith(("MT-", "mt-", "ERCC"))]]
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata = adata[:, adata.var.highly_variable]
            sc.pp.scale(adata, max_value=10)
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            st.session_state["adata"] = adata
            st.success("✅ Preprocessing complete")
    else:
        st.warning("⚠️ Please upload a .h5ad file first")

# Tab 3 - ML
with tabs[2]:
    st.header("🤖 ML Pipeline")
    df = st.session_state.get("df", None)
    if df is not None:
        target = st.selectbox("Select target column", df.columns)
        test_size = st.slider("Test size", 0.1, 0.5, 0.3)
        features = [c for c in df.columns if c != target]
        X = pd.get_dummies(df[features])
        y = df[target]
        task = "classification" if y.dtype == "object" or y.nunique() < 15 else "regression"
        if task == "classification":
            y = LabelEncoder().fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        model = RandomForestClassifier() if task == "classification" else RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        plt.tight_layout(pad=1.0)
        if task == "classification":
            acc = accuracy_score(y_test, preds)
            st.success(f"Accuracy: {acc:.2f}")
            sns.heatmap(confusion_matrix(y_test, preds), annot=True, cmap="Blues", fmt="d", ax=ax)
        else:
            r2 = r2_score(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            st.success(f"R²: {r2:.2f}")
            st.info(f"MSE: {mse:.2f}")
            ax.scatter(y_test, preds, alpha=0.5)
            ax.set_title("Predicted vs Actual")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Upload a CSV file first")

# Tab 4 - DEG + Volcano
with tabs[3]:
    st.header("🧬 DEG + Volcano Plot")
    adata = st.session_state.get("adata", None)
    if adata is not None:
        categorical_cols = [col for col in adata.obs.columns if adata.obs[col].dtype.name == "category" or adata.obs[col].nunique() < 30]
        if not categorical_cols:
            st.error("❌ No categorical metadata found in AnnData.obs")
        else:
            groupby = "celltype" if "celltype" in adata.obs.columns else categorical_cols[0]
            groups = list(adata.obs[groupby].unique())
            if len(groups) < 2:
                st.error("❌ Need at least 2 groups for DEG")
            else:
                pairs = list(itertools.combinations(groups, 2))[:10]
                pair_labels = [f"{a} vs {b}" for a, b in pairs]
                selected_pair = st.selectbox("Choose Group Comparison", pair_labels)
                test_group, reference = selected_pair.split(" vs ")
                method = st.radio("Method", ["t-test", "wilcoxon"])

                if st.button("Run DEG + Volcano Plot"):
                    try:
                        sc.tl.rank_genes_groups(
                            adata,
                            groupby=groupby,
                            method=method,
                            groups=[test_group],
                            reference=reference,
                            use_raw=False
                        )
                        df_deg = sc.get.rank_genes_groups_df(adata, group=test_group).dropna()
                        if df_deg.empty:
                            st.error("❌ DEG returned no results")
                        else:
                            df_deg["neg_log10_pval"] = -np.log10(df_deg["pvals"].replace(0, np.nan))
                            df_deg["diffexpressed"] = "NS"
                            df_deg.loc[(df_deg["logfoldchanges"] > 1) & (df_deg["pvals_adj"] < 0.05), "diffexpressed"] = "UP"
                            df_deg.loc[(df_deg["logfoldchanges"] < -1) & (df_deg["pvals_adj"] < 0.05), "diffexpressed"] = "DOWN"

                            fig, ax = plt.subplots(figsize=(12, 5))
                            sns.scatterplot(
                                data=df_deg,
                                x="logfoldchanges",
                                y="neg_log10_pval",
                                hue="diffexpressed",
                                palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "gray"},
                                ax=ax,
                                alpha=0.7
                            )
                            ax.axhline(-np.log10(0.05), color='gray', linestyle='--')
                            ax.axvline(1, color='gray', linestyle='--')
                            ax.axvline(-1, color='gray', linestyle='--')
                            ax.set_xlim(-5, 5)
                            ax.set_title(f"Volcano Plot: {test_group} vs {reference}")
                            st.pyplot(fig)
                            st.dataframe(df_deg.head(10))
                    except Exception as e:
                        st.error(f"❌ DEG/Volcano failed: {e}")
    else:
        st.warning("⚠️ Upload and preprocess a .h5ad file first")

# Tab 5 - Expression Plot
with tabs[4]:
    st.header("📊 Expression Visualization")
    adata = st.session_state.get("adata", None)
    if adata is not None:
        gene = st.text_input("Enter gene name")
        if gene:
            if gene not in adata.var_names:
                st.error("Gene not found in dataset")
            else:
                if "neighbors" not in adata.uns:
                    sc.pp.neighbors(adata)
                if "X_umap" not in adata.obsm:
                    sc.tl.umap(adata)

                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots()
                    sc.pl.violin(adata, gene, groupby=adata.obs.columns[0], show=False, ax=ax1)
                    st.pyplot(fig1)
                with col2:
                    fig2, ax2 = plt.subplots()
                    sc.pl.umap(adata, color=gene, show=False, ax=ax2)
                    st.pyplot(fig2)

# Tab 6 - Team
with tabs[5]:
    st.header("👥 Team")
    st.markdown("- George – ML Developer")
