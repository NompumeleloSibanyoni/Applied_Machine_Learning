from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Paths ---
DATA_PATH = Path("data/Percept1.xlsx")      # <-- put your file in repo /data
SHEET = "Sheet1"                             # change if needed

# --- Load data ---
# If your sheet already has headers, set header=0 and remove 'names'
df = pd.read_excel(DATA_PATH, sheet_name=SHEET, header=None, names=["feature1", "feature2", "target"])

# --- Basic cleaning (optional, safe) ---
df = df.dropna().reset_index(drop=True)

X = df[["feature1", "feature2"]]
y = df["target"]

# --- Train/test split (stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Pipeline: scale -> perceptron ---
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", Perceptron(max_iter=1000, tol=1e-3, random_state=42))
])

# --- Fit & evaluate ---
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.3f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))
