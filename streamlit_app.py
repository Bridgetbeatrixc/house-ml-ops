from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from src.data.data_loader import FEATURE_COLUMNS, load_data
from src.data.data_preprocessor import build_preprocessor, split_features_target
from src.models.model import create_model
from src.models.trainer import evaluate_model
from src.utils.config import load_config, resolve_path


st.set_page_config(
    page_title="My Portfolio with Streamlit",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_OPTIONS = {
    "Random Forest": (
        "random_forest",
        {
            "n_estimators": 120,
            "max_depth": 14,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
    ),
    "Gradient Boosting": (
        "gradient_boosting",
        {
            "n_estimators": 120,
            "learning_rate": 0.08,
            "max_depth": 3,
            "random_state": 42,
        },
    ),
    "Linear Regression": ("linear_regression", {}),
}


@st.cache_data(show_spinner=False)
def get_config() -> dict:
    return load_config()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    config = get_config()
    return load_data(
        resolve_path(config["data"]["path"]),
        target_column=config["data"]["target_column"],
        feature_columns=config["data"]["feature_columns"],
        random_state=config["project"]["random_state"],
    )


@st.cache_resource(show_spinner=False)
def get_prediction_model():
    config = get_config()
    model_path = resolve_path(config["artifacts"]["model_path"])
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def train_visual_model(model_label: str) -> tuple[Pipeline, dict[str, float]]:
    config = get_config()
    dataframe = get_dataset()
    model_type, params = MODEL_OPTIONS[model_label]
    split_data = split_features_target(
        dataframe,
        target_column=config["data"]["target_column"],
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_state"],
    )
    preprocessor = build_preprocessor(dataframe, config["data"]["target_column"])
    estimator = create_model(model_type, params)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(split_data.X_train, split_data.y_train)
    return pipeline, evaluate_model(pipeline, split_data.X_test, split_data.y_test)


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def plot_histogram(dataframe: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dataframe[column].dropna(), bins=28, color="#0f766e", edgecolor="white")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_correlation(dataframe: pd.DataFrame):
    numeric_frame = dataframe.select_dtypes(include="number")
    correlation = numeric_frame.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation, cmap="viridis")
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(correlation.columns)))
    ax.set_yticklabels(correlation.columns, fontsize=8)
    ax.set_title("Numeric Feature Correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def show_header() -> None:
    st.title("My Portfolio with Streamlit")
    st.markdown(
        """
        Portfolio ini menampilkan ringkasan profil, daftar proyek data science,
        visualisasi dataset House Prices dari Kaggle, dan implementasi prediksi
        harga rumah menggunakan pipeline machine learning yang sudah dilatih.
        """
    )


def show_about() -> None:
    st.subheader("Tentang Saya")
    col_left, col_right = st.columns([1.25, 1])
    with col_left:
        st.markdown(
            """
            **Nama:** Bridget Beatrix C.

            Saya sedang membangun portfolio data science dengan fokus pada
            machine learning workflow, model deployment, dan aplikasi data
            interaktif. Project ini memakai dataset House Prices Advanced
            Regression Techniques untuk menunjukkan proses dari data loading,
            preprocessing, training, tracking, hingga serving model.
            """
        )
    with col_right:
        st.markdown("**Keahlian utama**")
        st.write("- Python, Pandas, Scikit-learn")
        st.write("- Streamlit, FastAPI, Docker")
        st.write("- MLflow experiment tracking")
        st.write("- Data visualization dan model evaluation")


def show_projects() -> None:
    st.subheader("Proyek Saya")
    projects = [
        {
            "title": "House Price Prediction MLOps",
            "description": "Pipeline prediksi harga rumah dengan preprocessing, MLflow, FastAPI, Docker, monitoring, dan Streamlit.",
            "score": 92,
        },
        {
            "title": "Customer Analytics Dashboard",
            "description": "Dashboard analitik untuk segmentasi customer, tren transaksi, dan insight bisnis berbasis data.",
            "score": 84,
        },
        {
            "title": "Sales Forecasting Experiment",
            "description": "Eksperimen forecasting penjualan dengan evaluasi metrik dan visualisasi performa model.",
            "score": 78,
        },
    ]
    cols = st.columns(3)
    for col, project in zip(cols, projects):
        with col:
            st.markdown(f"### {project['title']}")
            st.write(project["description"])
            st.progress(project["score"] / 100, text=f"Completion: {project['score']}%")
            st.button("View Project", key=f"project-{project['title']}", disabled=True)


def portfolio_page() -> None:
    show_header()
    show_about()
    st.divider()
    show_projects()


def prediction_page() -> None:
    st.header("Prediksi Harga Rumah dari CSV")
    st.write(
        "Upload file CSV dengan kolom fitur Kaggle yang digunakan model. "
        "Klik tombol prediksi untuk menghasilkan kolom `PredictedSalePrice`."
    )
    st.caption(", ".join(FEATURE_COLUMNS))

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    model = get_prediction_model()
    if model is None:
        st.error("Model artifact belum ditemukan. Jalankan `python -m src.models.trainer` terlebih dahulu.")
        return

    if uploaded_file is not None:
        input_frame = pd.read_csv(uploaded_file)
        st.write("Preview data upload")
        st.dataframe(input_frame.head(20), use_container_width=True)

        missing_columns = [column for column in FEATURE_COLUMNS if column not in input_frame.columns]
        if missing_columns:
            st.error(f"Kolom berikut belum ada di file CSV: {missing_columns}")
            return

        if st.button("Run Prediction Pipeline", type="primary"):
            prediction_frame = input_frame[FEATURE_COLUMNS].copy()
            predictions = model.predict(prediction_frame)
            result_frame = input_frame.copy()
            result_frame["PredictedSalePrice"] = predictions

            st.success("Prediksi selesai.")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Rows Predicted", f"{len(result_frame):,}")
            col_b.metric("Average Prediction", format_currency(float(predictions.mean())))
            col_c.metric("Highest Prediction", format_currency(float(predictions.max())))
            st.dataframe(result_frame.head(100), use_container_width=True)
            st.download_button(
                "Download Prediction Results",
                data=result_frame.to_csv(index=False).encode("utf-8"),
                file_name="house_price_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload CSV untuk mulai prediksi.")


def data_visualization_page() -> None:
    st.header("Visualisasi Data")
    dataframe = get_dataset()
    target_column = get_config()["data"]["target_column"]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Rows", f"{len(dataframe):,}")
    col_b.metric("Features", f"{len(dataframe.columns) - 1:,}")
    col_c.metric("Target", target_column)

    st.dataframe(dataframe.head(30), use_container_width=True)

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    selected_column = st.selectbox(
        "Pilih fitur untuk distribusi",
        options=numeric_columns,
        index=numeric_columns.index(target_column) if target_column in numeric_columns else 0,
    )
    st.pyplot(plot_histogram(dataframe, selected_column))

    with st.expander("Correlation Heatmap", expanded=True):
        st.pyplot(plot_correlation(dataframe))


def model_performance_page() -> None:
    st.header("Visualisasi Performa Model")
    model_label = st.selectbox("Pilih model untuk divisualisasikan", list(MODEL_OPTIONS.keys()))

    with st.spinner(f"Training dan evaluasi {model_label}..."):
        _, metrics = train_visual_model(model_label)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("RMSE", f"{metrics['rmse']:,.2f}")
    col_b.metric("MAE", f"{metrics['mae']:,.2f}")
    col_c.metric("R2", f"{metrics['r2']:.3f}")

    metric_frame = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "R2"],
            "Value": [metrics["rmse"], metrics["mae"], metrics["r2"]],
        }
    )
    st.bar_chart(metric_frame.set_index("Metric"))

    st.info(
        "Karena ini regression problem, performa model ditampilkan dengan RMSE, MAE, dan R2, "
        "bukan confusion matrix yang biasanya dipakai untuk classification."
    )


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Pilih halaman",
        [
            "Portfolio",
            "CSV Prediction",
            "Data Visualization",
            "Model Performance",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption("House Prices Advanced Regression Techniques")

    if page == "Portfolio":
        portfolio_page()
    elif page == "CSV Prediction":
        prediction_page()
    elif page == "Data Visualization":
        data_visualization_page()
    else:
        model_performance_page()


if __name__ == "__main__":
    main()
