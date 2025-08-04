"""
Main script to test the Real Estate Price Prediction pipeline.
"""

from config.logger import get_logger
from preprocessing.preprocessing import load_data
from models.model import train_model, predict
from utils.helpers import format_price

logger = get_logger(__name__)

def main():
    try:
        logger.info("📁 Loading data...")
        df = load_data("data/final.csv")

        logger.info("🧹 Preparing features and target...")
        X = df.drop(columns=['price'])
        y = df['price']

        # Optional stratify column (if exists)
        stratify_col = X['property_type_Bunglow'] if 'property_type_Bunglow' in X.columns else None

        logger.info("🤖 Training model...")
        model, metrics, X_test, y_test = train_model(X, y, stratify_col)

        logger.info("✅ Model training complete.")
        logger.info(f"📊 Metrics: {metrics}")

        logger.info("🔍 Predicting a sample...")
        sample = X_test.iloc[[0]]
        prediction = predict(model, sample)
        formatted = format_price(prediction[0])
        logger.info(f"📌 Predicted price: {formatted}")

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()