import streamlit as st
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *

# Function to load a dataset from a file path
def load_dataset(file_path):
    return pd.read_csv(file_path)  # You can extend this to handle other data formats

# Function to detect column types (categorical or numerical)
def detect_column_types(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(exclude=['object']).columns
    return categorical_columns, numerical_columns

# Function to handle null values based on column type
def handle_null_values(df, categorical_columns, numerical_columns, categorical_strategy, numerical_strategy):
    for column in categorical_columns:
        if categorical_strategy == 'most_frequent':
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif categorical_strategy == 'additional_class':
            df[column].fillna("Missing", inplace=True)

    for column in numerical_columns:
        if numerical_strategy == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif numerical_strategy == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif numerical_strategy == 'mode':
            df[column].fillna(df[column].mode()[0], inplace=True)

# Function to detect the task type (classification or regression)
def detect_task_type(df, target_column):
    if df[target_column].nunique() <= 2:  # Assuming binary classification
        return "classification"
    else:
        return "regression"

# Function to perform modeling using PyCaret
def perform_modeling(df, target_column, task_type):
    if task_type == "classification":
        st.write("Performing Classification Task...")
        exp_clf = setup(df, target=target_column)
        best_model = compare_models(include=['lr', 'knn', 'dt', 'rf', 'xgboost', 'lightgbm', 'catboost'])
        final_model = tune_model(best_model)
        evaluate_model(final_model)
    else:
        st.write("Performing Regression Task...")
        exp_reg = setup(df, target=target_column)
        best_model = compare_models(include=['lr', 'ridge', 'lasso', 'en', 'dt', 'rf', 'xgboost', 'lightgbm', 'catboost'])
        final_model = tune_model(best_model)
        evaluate_model(final_model)

# Streamlit app
def main():
    st.title("Data Preprocessing and Modeling with PyCaret")
    st.sidebar.header("Configuration")

    file_path = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if file_path is not None:
        df = load_dataset(file_path)

        st.sidebar.write("Columns in the dataset:")
        st.sidebar.write(df.columns)

        target_column = st.sidebar.text_input("Enter the name of the target column:")

        if target_column:
            task_type = detect_task_type(df, target_column)

            st.sidebar.write(f"Detected task type: {task_type}")

            categorical_columns, numerical_columns = detect_column_types(df)

            # Controllers
            categorical_strategy = st.sidebar.selectbox("Choose strategy for categorical columns", ['most_frequent', 'additional_class'])
            numerical_strategy = st.sidebar.selectbox("Choose strategy for numerical columns", ['mean', 'median', 'mode'])

            if st.sidebar.button("Handle Null Values"):
                st.sidebar.write("Handling null values...")
                handle_null_values(df, categorical_columns, numerical_columns, categorical_strategy, numerical_strategy)

            if st.sidebar.button("Perform Modeling"):
                st.sidebar.write("Performing modeling...")
                perform_modeling(df, target_column, task_type)

if __name__ == "__main__":
    main()
