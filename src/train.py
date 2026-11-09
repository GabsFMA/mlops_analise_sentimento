import joblib
from pathlib import Path
import pandas as pd
from loguru import logger 

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT  = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "comentarios.csv"
OUTPUT_MODEL = ROOT / "models"

def load_data(data_path: Path) -> pd.DataFrame:
    """Load data from a CSV file."""
    df = pd.read_csv(DATA)
    return df["texto"].tolist(), df['sentimento'].tolist()

def main():
    X, y = load_data(DATA)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Save the model
    joblib.dump(pipeline, OUTPUT_MODEL / "sentiment_model.joblib")
    
    logger.success(f"\nModel saved to {OUTPUT_MODEL / 'sentiment_model.joblib'}")
    
if __name__ == "__main__":
    main()
