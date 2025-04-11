from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import MODEL_PATH

# Column metadata
ordinal_cols = ['experience_level']  # example
nominal_cols = ['department', 'location']  # example

df = load_data()
preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df, 'days_until_event', ordinal_cols, nominal_cols)
model = train_model(preprocessor, X_train, y_train, MODEL_PATH)
evaluate_model(model, X_test, y_test)