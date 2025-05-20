import pandas as pd
import statsmodels.api as sm

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
from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Visualisierung
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.figure_factory import create_scatterplotmatrix
from plotly.subplots import make_subplots

# Einlesen der Daten
df = pd.read_csv(r"C:\programmieren\code_source\Disease_symptom_and_patient_profile_dataset.csv")

#Grösse Datensatz
print(df.head)
print("Vorschau auf die ersten 5 Zeilen:")
print("\n Struktur des Datensatzes:")
print(df.info())


#Manuelle Daten löschen nicht nötig für unseren Datensatz

#Missing Variables
missing_values = df.isnull().mean() * 100
missing_values = missing_values[missing_values > 0]
missing_values = missing_values.sort_values(ascending=False)

if missing_values.empty:
    print("Es gibt keine fehlenden Werte im Datensatz.")
else:
    print("📋 Liste der fehlenden Variablen (in %):")
    print(missing_values)
    print(f"Anzahl Variablen mit fehlenden Werten: {len(missing_values)}")

### Daten Splitten###

# Zielvariable abtrennen
X = df.drop(columns=['Outcome Variable'])
y = df['Outcome Variable']

print(f"Die Form des Datensatzes mit den Trainingsvariablen ist: {X.shape}")
print(f"Die Form der Zielvariable ist: {y.shape}")


# Train/Test-Split (70/30 - besser als 80/20, da kleiner Datensatz und Kreuzvalidierung auch noch angewendet wird)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Daten abspeichern
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print(f"Die Anzahl der Zeilen und Spalten im Trainingsdatensatz ist: {X_train.shape}")
print(f"Die Anzahl der Zeilen und Spalten im Testdatensatz: {X_test.shape}")

# Kontrolle der Verteilung der Klassen
print(f"Die Klassenverteilung auf dem Train sieht folgendermassen aus: {y_train.value_counts()/len(y_train)}")
print(f"Die Klassenverteilung auf dem Test sieht folgendermassen aus: {y_test.value_counts()/len(y_test)}")

# Daten Formate - Kategorielle Variablen
print("\nDatentypen der Trainingsvariablen:")
print(X_train.dtypes)

# Welche Kategorien sind enthalten, wiviele verschiedene
object_columns = X_train.select_dtypes(include=['object']).columns
print(f"Liste von kategorischen Variablen: {object_columns}")
print(f"Wieviele kategorischen Variablen sind vorhanden: {len(object_columns)}")

# Kardinalitäten der kategorischen Varialben
for col in  X_train.select_dtypes(include=['object']):
    cardinality = len(pd.Index(X_train[col].value_counts()))
    print(X_train[col].name + ": " + str(cardinality))


### Datenaufbereitung ###

## One-Hot-Encoding
# Ausgangsformen vor dem Encoding
print(f"Shape of X_train before encoding: {X_train.shape}")
print(f"Shape of X_test before encoding: {X_test.shape}")

# One-Hot-Encoding mit k Dummies (drop_last=False)
encoder_k = OneHotEncoder(
    variables=['Disease', 'Blood Pressure', 'Cholesterol Level'],
    drop_last=False
)
encoder_k.fit(X_train)

# Transformieren mit k-Encoder
X_train_k = encoder_k.transform(X_train)
X_test_k = encoder_k.transform(X_test)

print(f"\nShape after k-Encoding (drop_last=False):")
print(f"X_train: {X_train_k.shape}, X_test: {X_test_k.shape}")

# One-Hot-Encoding mit k–1 Dummies (drop_last=True)
encoder_k_1 = OneHotEncoder(
    variables=['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender'],
    drop_last=True
)
encoder_k_1.fit(X_train_k)

# Transformieren mit k–1-Encoder
X_train_final = encoder_k_1.transform(X_train_k)
X_test_final = encoder_k_1.transform(X_test_k)

print(f"\nShape after additional k–1-Encoding (drop_last=True):")
print(f"X_train: {X_train_final.shape}, X_test: {X_test_final.shape}")

# Ausgabe der automatisch verarbeiteten Variablen
print("\nVariablen, die vom k-Encoder verarbeitet wurden:")
print(encoder_k.variables_)

print("Variablen, die vom k-1 Encoder verarbeitet wurden:")
print(encoder_k_1.variables_)

## Konstante und Quasi-Konstante
sel = DropConstantFeatures(tol=0.95, variables=None, missing_values='raise')

# quasi-constant drop as tol is not equal to 1, with tol = 1 we get Constant feature dropping
sel.fit(X_train_final)

# list of quasi-constant features
sel.features_to_drop_

# Umformung durchführen
X_train_final = sel.transform(X_train_final)
X_test_final = sel.transform(X_test_final)

print(f"Dimension des Train Datensatzes: {X_train_final.shape}")
print(f"Dimension des Test Datensatzes: {X_test_final.shape}")

## Analyse Duplikate
# Initialisierung des Selektors
sel = DropDuplicateFeatures(variables=None, missing_values='raise')

# Identification von Duplikaten
sel.fit(X_train_final)

# Hier würden die Duplikate angezeigt werden.
# each set are duplicates

print("\nGefundene Duplikatengruppen (jede Gruppe hat identische Spalten):")
print(sel.duplicated_feature_sets_)

# Hier werden alle Duplikatenpaare angezeigt.
# 1 from each of the pairs above

print("\nSpalten, die entfernt werden, weil sie Duplikate sind:")
print(sel.features_to_drop_)

## Korrelierende Werte
# correlation selector

sel = SmartCorrelatedSelection(
    variables=None,
    method="pearson",
    threshold=0.8,
    missing_values="raise",
    selection_method="variance",
    estimator=None,
    cv=1
)

sel.fit(X_train_final)
print("\nKorrelierende Merkmalsgruppen (r > 0.8):")
print(sel.correlated_feature_sets_)

# welche Variable zum Löschen vorgeschlagen wird
print("\nVorgeschlagene Merkmale zum Entfernen:")
print(sel.features_to_drop_)

#=== wir löschen keine, wir haben uns dazu entschieden alle beizubehalten
# Weil wir nicht löschen, muss auch keine Umformung durchgeführt werden

## Variablen speichern
X_train_final.to_csv(r"C:\programmieren\v_studio_c\X_train_disease_final_encoded.csv", index=False)
print("Trainingsdaten wurden erfolgreich gespeichert.")

X_test_final.to_csv(r"C:\programmieren\v_studio_c\X_test_disease_final_encoded.csv", index=False)
print("Testdaten wurden erfolgreich gespeichert.")

### Deskriptive Statistik ###
# Create a subplot for each feature
fig = make_subplots(rows=len(X_train.columns), cols=2, 
                    subplot_titles=[f"{col} (Train)" for col in X_train.columns] + [f"{col} (Test)" for col in X_test.columns])


# Add histograms for the training data in the first column
for i, column in enumerate(X_train.columns):
    fig.add_trace(
        go.Histogram(x=X_train[column], name=f"Train: {column}", 
                     histnorm='probability', marker_color='blue'),
        row=i+1, col=1
    )
    
# Add histograms for the test data in the second column
for i, column in enumerate(X_test.columns):
    fig.add_trace(
        go.Histogram(x=X_test[column], name=f"Test: {column}", 
                     histnorm='probability', marker_color='orange'),
        row=i+1, col=2
    )


# Update layout
fig.update_layout(height=300*len(X_train.columns), title_text="Histograms of All Variables")

# Show the plot
fig.show()
