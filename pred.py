import pickle
import pandas as pd

from pathlib import Path

from src import preprocess, transform
from src.PaintingClassifier import PaintingClassifier

# Default lightweight inference artifact path used by the submission script.
MODEL_FILE = Path("artifacts/baseline_logreg_tfidf.pkl")
    
def predict_all(filename):
    with MODEL_FILE.open("rb") as file:
        artifact = pickle.load(file)

    raw = pd.read_csv(filename)

    column_mapping = preprocess.resolve_columns(raw.columns,
                                                require_label=False)
    cleaned = preprocess.clean_dataframe(df=raw,
                                         column_mapping=column_mapping,
                                         require_label=False)

    x = transform.transform_features(df=cleaned,
                                    feature_state=artifact["feature_state"])
    
    model = PaintingClassifier.from_artifact(artifact)
    
    return model.predict(x).tolist()
