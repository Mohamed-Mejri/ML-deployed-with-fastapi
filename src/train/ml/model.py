import os
import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from constants import CAT_FEATURES

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForest model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def save_model(model, name="model.pkl", path="../../model/"):
    """ Saves `model` to `path` under `name`.

    Inputs
    ------
    model : RandomForest model
        Trained machine learning model.
    name : str
        model name.
    path : str
        path where to save the model.
    Returns
    -------
    None
    """
    full_path = os.path.join(path, name)
    pickle.dump(model, open(full_path, "wb"))

def performance_slices(X, y, preds, output="slice_output.txt"):
    df = X.copy()
    df["target"] = y
    df["preds"] = preds
    
    with open(os.path.join(os.getcwd(), output), "w") as fp:
        for cat_feat in CAT_FEATURES:
            fp.write(f"**** Slicing on the '{cat_feat}' feature ****\n")
            for cat_value in df[cat_feat].unique():
                fp.write(f"Value = {cat_value}\n")
                slice = df[df[cat_feat]==cat_value]
                precision, recall, fbeta = compute_model_metrics(y=slice["target"], preds=slice["preds"])
                metrics = f"precision: {precision} \t recall: {recall} \t f1_score: {fbeta} \n"
                fp.write(metrics)
