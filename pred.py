import pickle
import pandas as pd

from pathlib import Path

from src import preprocess, features

# The model .pkl file
MODEL_FILE = "artifacts/baseline_logreg_tfidf.pkl"
    
def predict_all(filename):

    with open(Path(MODEL_FILE), "rb") as file:
        artifact = pickle.load(file)

    with open(filename, "r", encoding="utf-8") as file:
        raw = pd.read_csv(file)
    
    model = artifact["model"]

    feature_state = {
        "vectorizer": artifact["vectorizer"],
        "fill_values": artifact["fill_values"],
        "text_columns": artifact["feature_config"]["text_columns"],
        "structured_columns": artifact["feature_config"]["structured_columns"],
        "categorical_columns": artifact["feature_config"].get("categorical_columns", []),
        "categorical_encoders": artifact["feature_config"].get("categorical_encoders", {}),
        "tfidf_config": artifact["feature_config"]["tfidf_config"],
    }

    column_mapping = preprocess.resolve_columns(raw.columns)
    cleaned = preprocess.clean_dataframe(df=raw,
                                         column_mapping=column_mapping)

    x = features.transform_features(df=cleaned,
                                    feature_state=feature_state)
    
    return model.predict(x)

if __name__ == "__main__":
    # Sanity check to ensure function correctly outputs without error
    # Delete before submission
    print(predict_all("data/training_data_202601.csv"))
    