# rainfall_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, log_loss

def load_data(file_path):
    """Load rainfall dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data if needed."""
    # Add preprocessing steps if required
    return df

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate the machine learning model."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        'Accuracy Score': accuracy_score(y_test, predictions),
        'Jaccard Index': jaccard_score(y_test, predictions),
        'F1-Score': f1_score(y_test, predictions)
    }

    if isinstance(model, SVC):
        metrics['LogLoss'] = log_loss(y_test, predictions)

    return metrics

def main():
    # Step 1: Load your rainfall dataset
    dataset_path = '../data/your_dataset.csv'  # Update the path accordingly
    rainfall_data = load_data(dataset_path)

    # Step 2: Preprocess the data (if needed)
    rainfall_data = preprocess_data(rainfall_data)

    # Step 3: Split the dataset into training and testing data for classification
    X = rainfall_data.drop('RainTomorrow', axis=1)
    y = rainfall_data['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4-7: Build and train models
    logistic_model = LogisticRegression()
    knn_model = KNeighborsClassifier(n_neighbors=3)
    tree_model = DecisionTreeClassifier()
    svm_model = SVC(probability=True)

    logistic_metrics = train_and_evaluate_model(logistic_model, X_train, X_test, y_train, y_test)
    knn_metrics = train_and_evaluate_model(knn_model, X_train, X_test, y_train, y_test)
    tree_metrics = train_and_evaluate_model(tree_model, X_train, X_test, y_train, y_test)
    svm_metrics = train_and_evaluate_model(svm_model, X_train, X_test, y_train, y_test)

    # Step 8: Create a final classification report/table of evaluation metrics
    classification_report = pd.DataFrame({
        'Metric': list(logistic_metrics.keys()),
        'Logistic Regression': list(logistic_metrics.values()),
        'KNN': list(knn_metrics.values()),
        'Decision Trees': list(tree_metrics.values()),
        'SVM': list(svm_metrics.values())
    })

    # Print or save the final classification report
    print(classification_report)

if __name__ == "__main__":
    main()
