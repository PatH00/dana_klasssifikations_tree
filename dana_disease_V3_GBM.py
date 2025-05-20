###  GBM - GradientBoostingClassifier ###

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Visualisierung
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from plotly.figure_factory import create_scatterplotmatrix
from plotly.subplots import make_subplots

### Einlesen der Daten ###
X_train = pd.read_csv(r"C:\programmieren\v_studio_c\X_train_disease_final_encoded.csv")
print(f"Shape of X_train data set {X_train.shape}")
X_test = pd.read_csv(r"C:\programmieren\v_studio_c\X_test_disease_final_encoded.csv")
print(f"Shape of X_test data set {X_test.shape}")
y_train = pd.read_csv("y_train.csv").squeeze()
print(f"Shape of y_train {y_train.shape}")
y_test = pd.read_csv("y_test.csv").squeeze()
print(f"Shape of y_test {y_test.shape}")

# Zielvariablen in numerische Werte umwandeln
y_train = y_train.map({"Negative": 0, "Positive": 1})
y_test = y_test.map({"Negative": 0, "Positive": 1})

# Optimierung der Hyperparameter
clf = GradientBoostingClassifier(
        min_samples_leaf=10,
        random_state = 0
)

clf = clf.fit(X_train, y_train)

### Kreuzvalidierung ###
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print("\nKreuzvalidierungsergebnisse (Accuracy pro Fold):", cv_scores)
print("Durchschnittliche Genauigkeit (CV):", round(cv_scores.mean(), 4))

# Varialbe Importance
# 'X_train' enthält die Spaltennamen des DataFrames
feat_imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=True)

# DataFrame für Plotly vorbereiten
feat_imp_df = feat_imp.reset_index()
feat_imp_df.columns = ['Feature', 'Importance']

# Plotly-Balkendiagramm (horizontal)
fig = px.bar(
    feat_imp_df,
    x='Importance',
    y='Feature',
    orientation='h',  # Horizontale Balken
    title='Importance of Features'
)

# Achsentitel anpassen
fig.update_layout(
    xaxis_title='Feature Importance Score',
    yaxis_title='Features',
    title_x=0.5  # Zentriert den Titel
)

# Grafik anzeigen
fig.show()

### Predict Variables ###
# Klassenverteilung für korrekte Interpretation der Ergebnisse
print(f"Anzahl pro binäre Variable im Train:")
print(y_train.value_counts(normalize=True))
print(f"Anzahl pro binäre Variable im Test:")
print(y_test.value_counts(normalize=True))

# Evaluierung der Modelle für Vorhersage für Trainings- und Testdaten
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

### Evaluationsmetriken - Konfusionsmatrix ###
# Train Confusion Matrix
confusion_matrix_train = metrics.confusion_matrix(y_train, y_pred_train)
print(f"Die Konfusionsmatrix des Training Sample ist:")
metrics.ConfusionMatrixDisplay(confusion_matrix_train).plot()
plt.show()

# Test Confusion Matrix
confusion_matrix_test = metrics.confusion_matrix(y_test, y_pred_test)
print(f"Die Konfusionsmatrix des Test Sample ist:")
metrics.ConfusionMatrixDisplay(confusion_matrix_test).plot()
plt.show()

### Accuracy-Score ###
# Verteilung der Klassen: 53% positiv zu 47% negativ
print("Accuracy Score im Trainingsset ist: {}".format(metrics.accuracy_score(y_train, y_pred_train)))
print("Accuracy Score im Testset ist: {}".format(metrics.accuracy_score(y_test, y_pred_test)))

### F1-Score ###
# kombiniert Precision und Recall
print("F1 Score im Trainingsset ist:", f1_score(y_train, y_pred_train))
print("F1 Score im Testset ist:", f1_score(y_test, y_pred_test))

### Area under the Curve AUC ###
# kombiniert die Performance der Klassen 0 und 1
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_train)
print("AUC score for the training set: {}".format(metrics.auc(fpr, tpr)))
# AUC for test
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_test)
print("AUC score for the test set: {}".format(metrics.auc(fpr, tpr)))
