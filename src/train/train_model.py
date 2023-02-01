# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import load_data_s3, process_data
from constants import CAT_FEATURES
from ml.model import train_model, inference, compute_model_metrics, save_model
# Add the necessary imports for the starter code.

data = load_data_s3()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=CAT_FEATURES, label="salary", training=True
)

# Train and save a model.
model = train_model(X_train, y_train)
save_model(model)
save_model(encoder, name="encoder.pkl")