import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Step 1: Data Processing - Read data from CSV file into a DataFrame
file_path = 'project_1_Data.csv'
df = pd.read_csv(file_path)

# Step 2: Data Visualization - Perform Statistical Analysis
print("Descriptive Statistics by Step:")
grouped_stats = df.groupby('Step').describe()
print(grouped_stats)

# Step 2 (continued): Data Visualization - Correlation Analysis & Visualizations
# Correlation matrix to understand relationships between features
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualization of Dataset Behaviour within each 'Step'
plt.figure(figsize=(18, 12))
# Boxplots for each feature by 'Step'
for idx, column in enumerate(['X', 'Y', 'Z'], start=1):
    plt.subplot(2, 3, idx)
    sns.boxplot(x='Step', y=column, data=df)
    plt.title(f'Boxplot of {column} by Step')

# Violin plots for each feature by 'Step'
for idx, column in enumerate(['X', 'Y', 'Z'], start=4):
    plt.subplot(2, 3, idx)
    sns.violinplot(x='Step', y=column, data=df)
    plt.title(f'Violin Plot of {column} by Step')

plt.tight_layout()
plt.show()

# Heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Heatmap of Feature Correlations')
plt.show()

# Step 3: Correlation Analysis - Transform Step into Target Variable
target_variable = 'Step'

# Step 4: Classification Model Development/Engineering - Prepare data and create models
X = df.drop(columns=[target_variable])
y = df[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the models and their hyperparameters for GridSearchCV
models = {
    'SVM': SVC(),
    'RandomForest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

param_grids = {
    'SVM': {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'class_weight': [None, 'balanced']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
}

# Perform GridSearchCV for each model
best_estimators = {}
for model_name in models:
    print(f"\nPerforming GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_estimators[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

# Step 5: Model Performance Analysis - Evaluate the best models on the test set
for model_name, model in best_estimators.items():
    y_pred = model.predict(X_test)
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

# Step 4 (continued): RandomizedSearchCV for RandomForestClassifier
random_search_params = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'class_weight': [None, 'balanced']
}

print("\nPerforming RandomizedSearchCV for RandomForest...")
random_search = RandomizedSearchCV(RandomForestClassifier(), random_search_params, n_iter=50, cv=5, n_jobs=-1, verbose=1, random_state=42)
random_search.fit(X_train, y_train)
best_random_forest = random_search.best_estimator_
print(f"\nBest parameters for RandomForest (RandomizedSearchCV): {random_search.best_params_}")

# Evaluate the RandomizedSearchCV model on the test set
y_pred_random = best_random_forest.predict(X_test)
print("\nClassification Report for RandomForest (RandomizedSearchCV):")
print(classification_report(y_test, y_pred_random))

# Step 6: Stacked Model Performance Analysis - Combine models using StackingClassifier
estimators = [
    ('svm', best_estimators['SVM']),
    ('knn', best_estimators['KNN'])
]

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(random_state=42))

# Train the stacking model
print("\nTraining Stacking Classifier...")
stacking_clf.fit(X_train, y_train)

# Evaluate the stacking model
y_pred_stacking = stacking_clf.predict(X_test)
print("\nClassification Report for Stacking Classifier:")
print(classification_report(y_test, y_pred_stacking))

# Use cross-validation for StackingClassifier
cross_val_scores = cross_val_score(stacking_clf, X_train, y_train, cv=5)
print("\nCross-Validation Scores for Stacking Classifier:", cross_val_scores)
print("Mean CV Score for Stacking Classifier:", cross_val_scores.mean())

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, f1

# Evaluate each model
model_performance = {}
for model_name, model in best_estimators.items():
    accuracy, precision, f1 = evaluate_model(model, X_test, y_test)
    model_performance[model_name] = {'accuracy': accuracy, 'precision': precision, 'f1': f1}

# Evaluate the stacking model
accuracy_stacking, precision_stacking, f1_stacking = evaluate_model(stacking_clf, X_test, y_test)
model_performance['Stacking'] = {'accuracy': accuracy_stacking, 'precision': precision_stacking, 'f1': f1_stacking}

# Evaluate the RandomizedSearchCV RandomForest model
accuracy_random, precision_random, f1_random = evaluate_model(best_random_forest, X_test, y_test)
model_performance['RandomForest (RandomizedSearchCV)'] = {'accuracy': accuracy_random, 'precision': precision_random, 'f1': f1_random}

# Step 5 (continued): Compare the performance metrics
print("\nModel Performance Comparison:")
for model_name, metrics in model_performance.items():
    print(f"\n{model_name}:")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")

# Create a confusion matrix for the stacking model
y_pred_best = stacking_clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Stacking Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 7: Model Evaluation - Save the selected model to a file
joblib.dump(stacking_clf, 'stacking_model.joblib')
print("\nStacking model saved to stacking_model.joblib")

# Load and use the model for predictions on new data
loaded_model = joblib.load('stacking_model.joblib')
new_data = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]
new_data_scaled = scaler.transform(new_data)
predictions = loaded_model.predict(new_data_scaled)

print("\nPredicted Maintenance Steps for New Data Points:")
for i, prediction in enumerate(predictions):
    print(f"Data Point {i+1}: {prediction}")