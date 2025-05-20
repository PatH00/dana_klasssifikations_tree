import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Feature Engine
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer,
)
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder, CountFrequencyEncoder
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
from feature_engine.selection import DropCorrelatedFeatures, SmartCorrelatedSelection
from feature_engine import transformation as vt
from feature_engine.wrappers import SklearnTransformerWrapper


# Scikit-Learn - Visualisation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Visualisierung
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.figure_factory import create_scatterplotmatrix
from plotly.subplots import make_subplots

### Einlesen der Daten ###
X_train = pd.read_csv(r"C:\programmieren\v_studio_c\X_train_disease_final_encoded.csv")
X_test = pd.read_csv(r"C:\programmieren\v_studio_c\X_test_disease_final_encoded.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

### Perspektive Statistik ###
print("Trainingsdatensatzgrösse:", X_train.shape)

# Fehlende Werte
missing_values = X_train.isnull().mean() * 100
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("\nAnteil fehlender Werte pro Variable (%):")
print(missing_values if not missing_values.empty else "Keine fehlenden Werte.")

### Modelltraining: Decision Tree mit GridSearchCV ###
param_grid = {
    'max_depth': [3, 5, 7],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10]
}

clf = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
clf.fit(X_train, y_train)

# === Evaluation ===
y_pred = clf.predict(X_test)

# ROC-AUC nur bei binärer Klassifikation
if len(clf.classes_) == 2:
    y_proba = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
else:
    roc_auc = "Nicht berechenbar – mehr als 2 Klassen"

# === Metriken ausgeben ===
print("Beste Parameter:", clf.best_params_)
print("\nKlassifikationsmetriken:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='macro'):.2f}")
print(f"ROC-AUC: {roc_auc}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# === Feature Importance ===
feature_importance = pd.Series(clf.best_estimator_.feature_importances_, index=X_train.columns)
print("\nTop 5 Merkmale:")
print(feature_importance.sort_values(ascending=False).head(5))

# === Visualisierung ===
feature_importance.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top 10 wichtigsten Merkmale")
plt.xlabel("Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
