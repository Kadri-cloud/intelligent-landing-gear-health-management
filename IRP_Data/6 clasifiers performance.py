import pandas as pd
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check if the results directory exists, if not, create it
if not os.path.exists("results"):
    os.makedirs("results")

# Load the datasets
data = pd.read_csv("MasterData.csv")
data_validator = pd.read_csv("MasterDataValidator.csv")

# Features and Target Variable
features = ['pumpSpeed', 'temp', 'pump_pressure', 'main_act_pressure_1', 'main_col_angle_1']
X = data[features]
y = data['health']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load validation data
X_valid = data_validator[features]
y_valid = data_validator['health']

# Convert string labels to integers for XGBoost compatibility
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_valid = label_encoder.transform(y_valid)

# Preprocessing with RobustScaler
scaler = RobustScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_valid_scaled = scaler.transform(X_valid)

# Polynomial features for Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
X_valid_poly = poly.transform(X_valid_scaled)

# Training the models
models = {
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=5000),
    "Polynomial Regression": LogisticRegression(solver='lbfgs', max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Train the models
for name, model in models.items():
    print(f"Training {name} Classifier...")
    if name == "Polynomial Regression":
        model.fit(X_train_poly, y_train)
    else:
        model.fit(X_train_scaled, y_train)

# Evaluate the models on test and validation datasets, and plot the results
for name, model in models.items():
    print(f"Evaluating {name} Classifier...")

    if name == "Polynomial Regression":
        y_pred_test = model.predict(X_test_poly)
        y_pred_valid = model.predict(X_valid_poly)
    else:
        y_pred_test = model.predict(X_test_scaled)
        y_pred_valid = model.predict(X_valid_scaled)

    # Accuracy
    test_accuracy = accuracy_score(y_test, y_pred_test)
    valid_accuracy = accuracy_score(y_valid, y_pred_valid)
    print(f"{name} Classifier Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"{name} Classifier Validation Accuracy: {valid_accuracy * 100:.2f}%\n")

    # Confusion matrices visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt="d", cmap="Blues", ax=ax[0], cbar=False)
    ax[0].set_title(f"{name} - Test Set")
    sns.heatmap(confusion_matrix(y_valid, y_pred_valid), annot=True, fmt="d", cmap="Blues", ax=ax[1], cbar=False)
    ax[1].set_title(f"{name} - Validation Set")
    plt.tight_layout()
    plt.savefig(f"results/{name}_confusion_matrices.png", dpi=300)
    plt.show()
