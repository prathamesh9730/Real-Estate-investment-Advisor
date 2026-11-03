import os
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "merged_houseprice_regression_classification-Regression+Class.csv"
CLF_PATH = APP_DIR / "best_classifier.pkl"
REG_PATH = APP_DIR / "regressor.pkl"
CURRENCY = "₹"
CURRENT_YEAR = 2026
 

GOOD_LABELS = {"good", "good investment", "high", "yes", "1", "true"}
 

MY_METRICS = {}
 

BASE_FEATURES = [
    "Area_sqft", "Bedrooms", "Bathrooms", "Floors", "Year_Built",
    "Location_Score", "School_Score", "Current_Price",
]
 
st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
 
 

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    df["Growth_Pct"] = (df["Price_After_5_Years"] - df["Current_Price"]) / df["Current_Price"] * 100
    df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]
    return df
 
 
@st.cache_resource
def load_models(df: pd.DataFrame):
    """Return (classifier, regressor, info dict)."""
    info = {"fallback_models": [], "metrics": {}}
    clf = joblib.load(CLF_PATH) if CLF_PATH.exists() else None
    reg = joblib.load(REG_PATH) if REG_PATH.exists() else None
    if clf is not None and reg is not None:
        return clf, reg, info

    X = df[BASE_FEATURES]
    Xtr, Xte, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, df["Price_Category"], df["Price_After_5_Years"], test_size=0.2, random_state=42
    )
    if clf is None:
        clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, yc_tr)
        info["fallback_models"].append("classifier")
        info["metrics"].update({
            "Accuracy": accuracy_score(yc_te, clf.predict(Xte)),
            "F1 (weighted)": f1_score(yc_te, clf.predict(Xte), average="weighted"),
        })
    if reg is None:
        reg = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr, yr_tr)
        info["fallback_models"].append("regressor")
        info["metrics"].update({
            "RMSE": float(np.sqrt(np.mean((yr_te - reg.predict(Xte)) ** 2))),
            "MAE": mean_absolute_error(yr_te, reg.predict(Xte)),
            "R²": r2_score(yr_te, reg.predict(Xte)),
        })
    return clf, reg, info
 
 
def model_columns(model):
    cols = getattr(model, "feature_names_in_", None)
    return list(cols) if cols is not None else BASE_FEATURES
 
 
def build_row(inputs: dict, cols: list, future_price: float | None = None) -> pd.DataFrame:
    row = dict(inputs)
    row["Age_of_Property"] = CURRENT_YEAR - inputs["Year_Built"]
    row["Price_per_SqFt"] = inputs["Current_Price"] / max(inputs["Area_sqft"], 1)
    if future_price is not None:
        current_price = inputs["Current_Price"]
        for year in range(1, 5):
            if current_price > 0 and future_price > 0:
                row[f"Price_Year_{year}"] = current_price * (future_price / current_price) ** (year / 5)
            else:
                row[f"Price_Year_{year}"] = current_price + (future_price - current_price) * year / 5
        row["Price_After_5_Years"] = future_price
    return pd.DataFrame([row]).reindex(columns=cols, fill_value=0)
 
 
def final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model
 
 
def importance_df(model, cols):
    est = final_estimator(model)
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_)
    elif hasattr(est, "coef_"):
        vals = np.abs(np.asarray(est.coef_)).reshape(-1, len(cols)).mean(axis=0) if np.size(est.coef_) % len(cols) == 0 else None
    else:
        vals = None
    if vals is None or len(vals) != len(cols):
        return None
    return pd.DataFrame({"Feature": cols, "Importance": vals}).sort_values("Importance")
 
 
def money(x: float) -> str:
    return f"{CURRENCY}{x:,.0f}"
 
 
# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: `{DATA_PATH}`. Put the CSV next to app.py or edit DATA_PATH.")
    st.stop()
 
df = load_data()
clf, reg, info = load_models(df)
category_labels = sorted(df["Price_Category"].astype(str).unique())
 
st.title("🏠 Real Estate Investment Advisor")
st.caption("Predict whether a property is a good investment and what it may cost after 5 years.")

tab_predict, tab_market, tab_model = st.tabs(["🔮 Predict", "📊 Market Insights", "🧠 Model Info"])
 
# ----------------------------------------------------------------------------
# TAB 1 - Prediction form
# ----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter property details")
    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        area = c1.number_input("Area (sqft)", 200, 20000, int(df["Area_sqft"].median()), step=50)
        bedrooms = c1.number_input("Bedrooms", 1, 10, int(df["Bedrooms"].median()))
        bathrooms = c1.number_input("Bathrooms", 1, 10, int(df["Bathrooms"].median()))
        floors = c2.number_input("Floors", 1, 10, int(df["Floors"].median()))
        year_built = c2.number_input("Year built", 1950, CURRENT_YEAR, int(df["Year_Built"].median()))
        current_price = c2.number_input(
            f"Current price ({CURRENCY})", 100000, 1_000_000_000, int(df["Current_Price"].median()), step=50000
        )
        loc_score = c3.slider("Location score", int(df["Location_Score"].min()), int(df["Location_Score"].max()),
                              int(df["Location_Score"].median()))
        school_score = c3.slider("School score", int(df["School_Score"].min()), int(df["School_Score"].max()),
                                 int(df["School_Score"].median()))
        submitted = st.form_submit_button("Analyse property", type="primary", width="stretch")
 
    if submitted:
        inputs = {
            "Area_sqft": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms, "Floors": floors,
            "Year_Built": year_built, "Location_Score": loc_score, "School_Score": school_score,
            "Current_Price": current_price,
        }
        try:
            row_r = build_row(inputs, model_columns(reg))
 
            future_price = float(reg.predict(row_r)[0])
            row_c = build_row(inputs, model_columns(clf), future_price)
            predicted_class = clf.predict(row_c)[0]
            model_classes = list(clf.classes_) if hasattr(clf, "classes_") else []
            encoded_labels = {
                int(value): category_labels[int(value)]
                for value in model_classes
                if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(category_labels)
            }
            label = encoded_labels.get(int(predicted_class), str(predicted_class))
            proba = clf.predict_proba(row_c)[0] if hasattr(clf, "predict_proba") else None
            classes = [encoded_labels.get(int(c), str(c)) for c in model_classes] if model_classes else None
        except Exception as e:  # feature mismatch with a custom model, etc.
            st.error(f"Prediction failed: {e}")
            st.info("Check that BASE_FEATURES in app.py match the columns your model was trained on.")
            st.stop()
 
        is_good = label.strip().lower() in GOOD_LABELS
        gain = future_price - current_price
        gain_pct = gain / current_price * 100
        cagr = ((future_price / current_price) ** (1 / 5) - 1) * 100 if future_price > 0 else 0
 
        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("#### Is this a Good Investment?")
            if is_good:
                st.success(f"✅ Yes - predicted category: **{label}**")
            else:
                st.warning(f"⚠️ Predicted category: **{label}**")
            if proba is not None:
                st.metric("Model confidence", f"{proba.max() * 100:.1f}%")
                st.plotly_chart(
                    px.bar(x=classes, y=proba * 100, labels={"x": "Category", "y": "Probability (%)"},
                           title="Class probabilities").update_layout(height=280),
                    width="stretch",
                )
        with r2:
            st.markdown("#### Estimated price after 5 years")
            st.metric("Future price", money(future_price), f"{gain_pct:+.1f}% ({money(gain)})")
            st.metric("Implied annual growth", f"{cagr:.2f}% / year")
 
            # projected path: smooth growth from current to predicted
            years = list(range(0, 6))
            path = [current_price * (future_price / current_price) ** (y / 5) for y in years]
            fig = go.Figure(go.Scatter(x=[f"Year {y}" for y in years], y=path, mode="lines+markers"))
            fig.update_layout(title="Projected price path", yaxis_title=f"Price ({CURRENCY})", height=280)
            st.plotly_chart(fig, width="stretch")
 
        st.markdown("#### Recommendation")
        market_growth = df["Growth_Pct"].median()
        if is_good and gain_pct >= market_growth:
            st.success(f"Strong candidate: expected growth of {gain_pct:.1f}% beats the dataset median of {market_growth:.1f}%.")
        elif is_good:
            st.info(f"Decent option: classified as good, but growth ({gain_pct:.1f}%) is below the dataset median ({market_growth:.1f}%).")
        else:
            st.warning(f"Think twice: not classified as a good investment. Expected growth {gain_pct:.1f}% vs median {market_growth:.1f}%.")
 
        imp = importance_df(clf, model_columns(clf))
        if imp is not None:
            st.plotly_chart(
                px.bar(imp.tail(8), x="Importance", y="Feature", orientation="h",
                       title="What drove the classification (top features)").update_layout(height=320),
                width="stretch",
            )
        st.caption("Predictions are model estimates for learning purposes, not financial advice.")
 
# ----------------------------------------------------------------------------
# TAB 2 - Market insights with filters
# ----------------------------------------------------------------------------
with tab_market:
    st.subheader("Filter properties")
    f1, f2, f3, f4 = st.columns(4)
    area_rng = f1.slider("Area (sqft)", int(df["Area_sqft"].min()), int(df["Area_sqft"].max()),
                         (int(df["Area_sqft"].min()), int(df["Area_sqft"].max())))
    price_rng = f2.slider("Current price", int(df["Current_Price"].min()), int(df["Current_Price"].max()),
                          (int(df["Current_Price"].min()), int(df["Current_Price"].max())))
    bhk = f3.multiselect("Bedrooms (BHK)", sorted(df["Bedrooms"].unique()), default=sorted(df["Bedrooms"].unique()))
    cats = f4.multiselect("Price category", sorted(df["Price_Category"].astype(str).unique()),
                          default=sorted(df["Price_Category"].astype(str).unique()))
 
    fdf = df[
        df["Area_sqft"].between(*area_rng)
        & df["Current_Price"].between(*price_rng)
        & df["Bedrooms"].isin(bhk)
        & df["Price_Category"].astype(str).isin(cats)
    ]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Properties", len(fdf))
    k2.metric("Avg current price", money(fdf["Current_Price"].mean()) if len(fdf) else "-")
    k3.metric("Avg price after 5 yrs", money(fdf["Price_After_5_Years"].mean()) if len(fdf) else "-")
    k4.metric("Median growth", f"{fdf['Growth_Pct'].median():.1f}%" if len(fdf) else "-")
 
    if fdf.empty:
        st.info("No properties match these filters.")
    else:
        a, b = st.columns(2)
        trend_cols = ["Current_Price", "Price_Year_1", "Price_Year_2", "Price_Year_3", "Price_Year_4", "Price_After_5_Years"]
        trend = fdf[trend_cols].mean().reset_index()
        trend.columns = ["Period", "Avg price"]
        trend["Period"] = ["Now", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
        a.plotly_chart(px.line(trend, x="Period", y="Avg price", markers=True, title="Average price trend (5 years)"),
                       width="stretch")
        b.plotly_chart(px.histogram(fdf, x="Current_Price", nbins=30, title="Distribution of current prices"),
                       width="stretch")
 
        c, d = st.columns(2)
        c.plotly_chart(px.scatter(fdf, x="Area_sqft", y="Current_Price", color="Price_Category",
                                  title="Area vs price"), width="stretch")
        d.plotly_chart(px.box(fdf, x="Bedrooms", y="Current_Price", title="Price by bedrooms (outliers visible)"),
                       width="stretch")
 
        e, f = st.columns(2)
        corr = fdf.select_dtypes("number").drop(columns=["Property_ID"], errors="ignore").corr()
        e.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                                 title="Correlation heatmap"), width="stretch")
        heat = fdf.assign(
            Loc=pd.cut(fdf["Location_Score"], 5), Sch=pd.cut(fdf["School_Score"], 5)
        ).pivot_table(index="Loc", columns="Sch", values="Current_Price", aggfunc="mean", observed=False)
        heat.index, heat.columns = heat.index.astype(str), heat.columns.astype(str)
        f.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Viridis",
                                 labels=dict(x="School score", y="Location score", color="Avg price"),
                                 title="Avg price: Location score × School score"), width="stretch")
 
        st.markdown("#### Matching properties")
        st.dataframe(fdf.drop(columns=["Age_of_Property"]), width="stretch", hide_index=True)
        st.download_button("Download filtered data (CSV)", fdf.to_csv(index=False), "filtered_properties.csv", "text/csv")
 
# ----------------------------------------------------------------------------
# TAB 3 - Model info
# ----------------------------------------------------------------------------
with tab_model:
    st.subheader("Model details")
    m = info["metrics"] or MY_METRICS
    if m:
        st.markdown("Held-out test metrics" + (" (demo models trained by the app):" if info["metrics"] else " (from MLflow):"))
        cols = st.columns(len(m))
        for col, (k, v) in zip(cols, m.items()):
            col.metric(k, f"{v:,.3f}")
    else:
        st.info("Fill the `MY_METRICS` dict at the top of app.py with your MLflow scores, or open the MLflow UI with `mlflow ui`.")
 
    g1, g2 = st.columns(2)
    for col, model, title in [(g1, clf, "Classifier feature importance"), (g2, reg, "Regressor feature importance")]:
        imp = importance_df(model, model_columns(model))
        if imp is not None:
            col.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title=title),
                             width="stretch")
        else:
            col.info(f"{title}: not available for this model type.")
 
    st.markdown(
        "**Notes:** Price_Year_1–4 are excluded from model inputs to avoid data leakage. "
        "The dataset is small (500 rows), so treat predictions as indicative."
    )
 