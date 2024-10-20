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

# Read data from CSV file into a DataFrame
file_path = 'project_1_Data.csv'
df = pd.read_csv(file_path)

# 1. Perform Statistical Analysis
print("Descriptive Statistics by Step:")
grouped_stats = df.groupby('Step').describe()
print(grouped_stats)

# Correlation matrix to understand relationships between features
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# 2. Visualization of Dataset Behaviour within each 'Step'
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

# 3. Transform Step into Target Variable
target_variable = 'Step'

# Prepare the data for ML models
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

# Evaluate the best models on the test set
for model_name, model in best_estimators.items():
    y_pred = model.predict(X_test)
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

# 4. RandomizedSearchCV for RandomForestClassifier
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

