import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"
DATA_PATH = ROOT / "data" / "train.csv"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    return json.loads(META_PATH.read_text(encoding="utf-8"))

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
meta = load_metadata()
df = load_data()

st.title("🏠 House Price Prediction")
st.write(
    "An end-to-end machine-learning application for predicting house sale prices "
    "from the Ames Housing / Kaggle House Prices dataset."
)

best = meta["best_model"]
metrics = meta["metrics"][best]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", best)
c2.metric("R²", f"{metrics['R2']:.3f}")
c3.metric("RMSE", f"${metrics['RMSE']:,.0f}")
c4.metric("MAE", f"${metrics['MAE']:,.0f}")

st.divider()

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Dataset", "ℹ️ Project"])

with tab1:
    st.subheader("Enter property information")
    st.caption("Fields are grouped so you can start with the most important property characteristics.")

    defaults = {
        "OverallQual": int(df["OverallQual"].median()),
        "GrLivArea": float(df["GrLivArea"].median()),
        "GarageCars": float(df["GarageCars"].median()),
        "TotalBsmtSF": float(df["TotalBsmtSF"].median()),
        "YearBuilt": int(df["YearBuilt"].median()),
        "FullBath": float(df["FullBath"].median()),
        "BedroomAbvGr": float(df["BedroomAbvGr"].median()),
        "KitchenAbvGr": float(df["KitchenAbvGr"].median()),
        "TotRmsAbvGrd": float(df["TotRmsAbvGrd"].median()),
        "Fireplaces": float(df["Fireplaces"].median()),
    }

    # Core features are the most useful starting point; advanced fields cover the entire dataset.
    core = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
            "YearBuilt", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd",
            "Fireplaces", "LotArea"]

    values = {}
    core_cols = st.columns(2)
    for i, feature in enumerate(core):
        if feature not in meta["all_features"]:
            continue
        with core_cols[i % 2]:
            if feature in meta["categorical_features"]:
                options = sorted(df[feature].dropna().astype(str).unique().tolist())
                values[feature] = st.selectbox(feature, options, key=f"core_{feature}")
            else:
                med = float(df[feature].median())
                if pd.api.types.is_integer_dtype(df[feature]):
                    values[feature] = st.number_input(
                        feature, value=int(defaults.get(feature, med)), step=1, key=f"core_{feature}"
                    )
                else:
                    values[feature] = st.number_input(
                        feature, value=float(defaults.get(feature, med)), step=1.0, key=f"core_{feature}"
                    )

    with st.expander("Advanced features — all remaining columns"):
        remaining = [x for x in meta["all_features"] if x not in core]
        adv_cols = st.columns(2)
        for i, feature in enumerate(remaining):
            with adv_cols[i % 2]:
                if feature in meta["categorical_features"]:
                    options = sorted(df[feature].dropna().astype(str).unique().tolist())
                    values[feature] = st.selectbox(
                        feature, options, index=0, key=f"adv_{feature}"
                    )
                else:
                    series = df[feature].dropna()
                    med = float(series.median()) if len(series) else 0.0
                    if pd.api.types.is_integer_dtype(df[feature]):
                        values[feature] = st.number_input(
                            feature, value=int(med), step=1, key=f"adv_{feature}"
                        )
                    else:
                        values[feature] = st.number_input(
                            feature, value=med, step=1.0, key=f"adv_{feature}"
                        )

    if st.button("Predict Sale Price", type="primary", use_container_width=True):
        input_df = pd.DataFrame([values], columns=meta["all_features"])
        prediction = float(model.predict(input_df)[0])
        st.success(f"Estimated Sale Price: ${prediction:,.0f}")
        st.info(
            "This is a machine-learning estimate, not a professional property valuation. "
            "Actual prices depend on location, market conditions, and property inspection."
        )

with tab2:
    st.subheader("Dataset overview")
    st.write(f"**{df.shape[0]:,} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Missing values")
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    st.dataframe(
        missing.rename("missing_count").to_frame(),
        use_container_width=True,
    )

    st.subheader("SalePrice distribution")
    st.bar_chart(df["SalePrice"].value_counts(bins=20).sort_index())

with tab3:
    st.subheader("What happens before prediction?")
    st.markdown("""
1. **Load data** from `data/train.csv`.
2. **Separate target** (`SalePrice`) from input features.
3. **Remove `Id`** because it is an identifier rather than a useful house characteristic.
4. **Handle missing numeric values** using median imputation.
5. **Handle missing categorical values** using most-frequent imputation.
6. **Convert categorical variables to numbers** using One-Hot Encoding.
7. **Train several regression models** and compare MAE, RMSE and R².
8. **Save the best pipeline** with Joblib.
9. **Use the saved pipeline in this Streamlit web interface**.

### Why a Pipeline?
The exact same preprocessing used during training is automatically applied to new user input. This prevents training/serving preprocessing mismatches.

### Models included
- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression
""")
