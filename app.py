import streamlit as st
import pandas as pd
from preprocessing.preprocessing import load_data
from models.model import train_model, predict
from utils.helpers import format_price
from config.logger import get_logger

logger = get_logger(__name__)

@st.cache_data
def get_dataset():
    df = load_data("data/final.csv")
    return df

@st.cache_resource
def train_model_cached(df):
    X = df.drop(columns=['price'])
    y = df['price']
    stratify_col = X['property_type_Bunglow'] if 'property_type_Bunglow' in X.columns else None
    model, metrics, _, _ = train_model(X, y, stratify_col)
    return model, metrics, X.columns

def main():
    st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")
    st.title("🏠 Real Estate Price Predictor")
    st.write("Adjust the features below to predict the house price.")

    df = get_dataset()
    model, metrics, feature_cols = train_model_cached(df)

    st.sidebar.header("📊 Model Performance")
    st.sidebar.metric("MAE", f"${metrics['MAE']:,.0f}")
    st.sidebar.metric("RMSE", f"${metrics['RMSE']:,.0f}")
    st.sidebar.metric("R²", f"{metrics['R2']:.2%}")

    # Create input form based on one example row
    input_data = {}
    sample = df.drop(columns=['price']).iloc[0]
    st.subheader("🏗 Input Features")

    for col in feature_cols:
        val = sample[col]
        if isinstance(val, (int, float)):
            input_data[col] = st.number_input(f"{col}", value=float(val))
        else:
            input_data[col] = st.text_input(f"{col}", value=str(val))

    if st.button("Predict Price"):
        try:
            input_df = pd.DataFrame([input_data])
            pred = predict(model, input_df)
            formatted_price = format_price(pred[0])
            st.success(f"💰 Estimated Price: {formatted_price}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            st.error("Prediction failed. Please check the input format.")

if __name__ == "__main__":
    main()