import pickle
import pandas as pd

from pathlib import Path

from src import preprocess, transform
from src.PaintingClassifier import PaintingClassifier

# The model .pkl file
MODEL_FILE = "artifacts/dummy.pkl"
    
def predict_all(filename):

    with open(Path(MODEL_FILE), "rb") as file:
        artifact = pickle.load(file)

    with open(filename, "r", encoding="utf-8") as file:
        raw = pd.read_csv(file)

    feature_state = {
        "vectorizer": artifact["vectorizer"],
        "fill_values": artifact["fill_values"],
        "text_columns": artifact["feature_config"]["text_columns"],
        "structured_columns": artifact["feature_config"]["structured_columns"],
        "categorical_columns": artifact["feature_config"].get("categorical_columns", []),
        "categorical_encoders": artifact["feature_config"].get("categorical_encoders", {}),
        "tfidf_config": artifact["feature_config"]["tfidf_config"],
    }

    column_mapping = preprocess.resolve_columns(raw.columns,
                                                require_label=False)
    cleaned = preprocess.clean_dataframe(df=raw,
                                         column_mapping=column_mapping,
                                         require_label=False)

    x = transform.transform_features(df=cleaned,
                                    feature_state=feature_state)
    
    model = PaintingClassifier()
    
    return model.predict(x)

if __name__ == "__main__":
    # Sanity check to ensure function correctly outputs without error
    # Delete before submission
    print(predict_all("data/training_data_202601_nolabel.csv"))
    