# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("CAD.csv")  # place CAD.csv in same directory
    # drop some columns with high correlation
    df.drop(['Weight', 'Length'], axis=1, inplace=True, errors="ignore")
    return df

df = load_data()
st.title("Heart Guard AI - CAD Prediction")
st.write("This app builds machine learning models to predict **Coronary Artery Disease (CAD)**")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Preprocessing
# -------------------------------
X = df.drop('Cath', axis=1)
y = df['Cath']

# Handle categorical/numerical features
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

num_cols = X.select_dtypes(include=['int64','float64']).columns
ord_cols = ['Region RWMA','Function Class']
nominal_cols = [col for col in X.select_dtypes(include=['object','bool']).columns if col not in ord_cols]

# order lists
Region_RWMA_order = [0,1,2,3,4]
Function_Class_order = [0,1,2,3]
cat_ord_list = [Region_RWMA_order, Function_Class_order]

# transformers
numerical_transformer = make_pipeline(StandardScaler())
nominal_transformer = make_pipeline(OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
ordinal_transformer = make_pipeline(OrdinalEncoder(categories=cat_ord_list))

preprocessor = make_column_transformer(
    (numerical_transformer, num_cols),
    (nominal_transformer, nominal_cols),
    (ordinal_transformer, ord_cols)
)

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=42)

# -------------------------------
# Train Models
# -------------------------------
# Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Decision Tree
dec_tree = DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train, y_train)
dec_pred = dec_tree.predict(X_test)
dec_acc = accuracy_score(y_test, dec_pred)

# KNN
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

# -------------------------------
# Display Results
# -------------------------------
st.subheader("Model Performance")
st.write(f"**Decision Tree Accuracy:** {dec_acc:.2f}")
st.write(f"**KNN Accuracy:** {knn_acc:.2f}")

col1, col2 = st.columns(2)
with col1:
    st.write("Confusion Matrix - Decision Tree")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, dec_pred, cmap="Blues", ax=ax)
    st.pyplot(fig)

with col2:
    st.write("Confusion Matrix - KNN")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, knn_pred, cmap="Reds", ax=ax)
    st.pyplot(fig)

# -------------------------------
# Decision Tree Visualization
# -------------------------------
st.subheader("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(15,8))
plot_tree(dec_tree, filled=True, feature_names=preprocessor.get_feature_names_out(), class_names=['No CAD','CAD'], fontsize=6)
st.pyplot(fig)

# -------------------------------
# Feature Importance
# -------------------------------
st.subheader("Feature Importance (Random Forest)")
importances = pd.DataFrame({
    'feature': preprocessor.get_feature_names_out(),
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

st.dataframe(importances.head(15))
