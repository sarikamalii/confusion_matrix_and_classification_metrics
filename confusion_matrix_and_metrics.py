import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

# Function to calculate metrics manually
def calculate_metrics_manually(actual, predicted):
    TP = sum((a == 1) and (p == 1) for a, p in zip(actual, predicted))
    FP = sum((a == 0) and (p == 1) for a, p in zip(actual, predicted))
    FN = sum((a == 1) and (p == 0) for a, p in zip(actual, predicted))
    TN = sum((a == 0) and (p == 0) for a, p in zip(actual, predicted))
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / len(actual)
    misclassification = 1 - accuracy
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    
    return precision, recall, accuracy, misclassification, f1_score

# Streamlit app for displaying a confusion matrix and metrics

st.title("Confusion Matrix and Metrics")

# Text area for inputting actual and predicted values
st.write("Input Actual and Predicted Values")
actual_values = st.text_area("Enter actual values (comma-separated):")
predicted_values = st.text_area("Enter predicted values (comma-separated):")

if actual_values and predicted_values:
    # Convert input strings to lists
    actual_values = [int(x.strip()) for x in actual_values.split(",")]
    predicted_values = [int(x.strip()) for x in predicted_values.split(",")]

    if len(actual_values) != len(predicted_values):
        st.write("The number of actual values and predicted values must be the same.")
    else:
        # Calculate metrics manually
        precision, recall, accuracy, misclassification, f1_score = calculate_metrics_manually(actual_values, predicted_values)

        # Calculate confusion matrix
        cm = confusion_matrix(actual_values, predicted_values)

        # Define labels for confusion matrix
        labels = ["Actual 0", "Actual 1"]
        predictions = ["Predicted 0", "Predicted 1"]
        
        # Display confusion matrix with labels
        cm_df = pd.DataFrame(cm, index=labels, columns=predictions)
        st.write("Confusion Matrix")
        st.write(cm_df)

        # Display metrics
        st.write("Metrics")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Misclassification Rate: {misclassification:.2f}")
        st.write(f"F1 Score: {f1_score:.2f}")

else:
    st.write("Please enter both actual and predicted values to proceed.")
