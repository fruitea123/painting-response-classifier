import pickle
import pandas as pd

from pathlib import Path

from src import preprocess, transform
from src.PaintingClassifier import PaintingClassifierLogreg, PaintingClassifierMNB

# Default lightweight inference artifact path used by the submission script.
MODEL_FILE = Path("artifacts/baseline_mnb_tfidf.pkl")


def load_lightweight_model(artifact):
    model_type = artifact["model_type"]
    model_state = artifact["model_state"]

    if model_state is None:
        raise ValueError(
            "pred.py requires a lightweight artifact, but the saved artifact only "
            "contains a sklearn model."
        )

    if model_type == "logreg":
        return PaintingClassifierLogreg.from_artifact(model_state)
    if model_type == "mnb":
        return PaintingClassifierMNB.from_artifact(model_state)
    raise ValueError(f"Unsupported model_type for pred.py: {model_type}")
    

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

    model = load_lightweight_model(artifact)

    return model.predict(x).tolist()
