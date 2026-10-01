import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("../Dataset/loan_data.csv")

print("Dataset loaded successfully")
print(df.head())

# ==========================================
# 2. Features and Target
# ==========================================

X = df.drop("loan_paid_back", axis=1)
y = df["loan_paid_back"]

# ==========================================
# 3. Identify Categorical Columns
# ==========================================

categorical_features = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade"
]

# ==========================================
# 4. Identify Numerical Columns
# ==========================================

numerical_features = [
    "age",
    "annual_income",
    "monthly_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "loan_term",
    "installment",
    "num_of_open_accounts",
    "total_credit_limit",
    "current_balance",
    "delinquency_history",
    "public_records",
    "num_of_delinquencies"
]

# ==========================================
# 5. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# ==========================================
# 6. Create ML Pipeline
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# ==========================================
# 7. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 8. Train Model
# ==========================================

model.fit(X_train, y_train)

print("Model trained successfully")

# ==========================================
# 9. Evaluate Model
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# ==========================================
# 10. Save Model
# ==========================================

joblib.dump(model, "model_joblib.pkl")

print("Model saved successfully!")
print("Saved as: model_joblib.pkl")