import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Read data from CSV file into a DataFrame
file_path = 'project_1_Data.csv'
df = pd.read_csv(file_path)

# 1. Perform Statistical Analysis
# Descriptive statistics for each feature grouped by 'Step'
print("Descriptive Statistics by Step:")
grouped_stats = df.groupby('Step').describe()
print(grouped_stats)

# Correlation matrix to understand relationships between features
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# 2. Visualization of Dataset Behaviour within each 'Step'

# Create a figure for all boxplots and violin plots together
plt.figure(figsize=(18, 12))

# Boxplots for each feature by 'Step'
for idx, column in enumerate(['X', 'Y', 'Z'], start=1):
    plt.subplot(2, 3, idx)
    sns.boxplot(x='Step', y=column, data=df)
    plt.title(f'Boxplot of {column} by Step')
    plt.xlabel('Step')
    plt.ylabel(column)

# Violin plots for each feature by 'Step'
for idx, column in enumerate(['X', 'Y', 'Z'], start=4):
    plt.subplot(2, 3, idx)
    sns.violinplot(x='Step', y=column, data=df)
    plt.title(f'Violin Plot of {column} by Step')
    plt.xlabel('Step')
    plt.ylabel(column)

# Adjust layout for better visibility
plt.tight_layout()
plt.show()

# Heatmap of the correlation matrix in a separate figure
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Heatmap of Feature Correlations')
plt.show()

# 3. Correlation with Target Variable
# Assuming 'Target' is the target variable in the dataset
target_variable = 'Target'

# Calculate the Pearson correlation between features and the target variable
correlation_with_target = df.corr()[target_variable].drop(target_variable)
print("\nCorrelation with Target Variable:")
print(correlation_with_target)

# Plot the correlation of features with the target variable
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values, palette='viridis')
plt.title('Correlation of Features with Target Variable')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.show()

# Explanation of the correlation
for feature, correlation in correlation_with_target.items():
    print(f"The feature '{feature}' has a correlation coefficient of {correlation:.2f} with the target variable.")
    if correlation > 0.5:
        print(f"This indicates a strong positive correlation, meaning as '{feature}' increases, the target variable tends to increase.")
    elif correlation < -0.5:
        print(f"This indicates a strong negative correlation, meaning as '{feature}' increases, the target variable tends to decrease.")
    else:
        print(f"This indicates a weak or moderate correlation with the target variable.")
        
        # Prepare the data
        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features
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
                'C': [0.1, 1, 10, 100],
                'gamma': [1, 0.1, 0.01, 0.001],
                'kernel': ['rbf', 'linear']
            },
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }

        # Perform GridSearchCV for each model
        best_estimators = {}
        for model_name in models:
            grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            best_estimators[model_name] = grid_search.best_estimator_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")

        # Evaluate the best models on the test set
        for model_name, model in best_estimators.items():
            y_pred = model.predict(X_test)
            print(f"\nClassification Report for {model_name}:")
            print(classification_report(y_test, y_pred))

        # RandomizedSearchCV for RandomForestClassifier
        random_search_params = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6]
        }

        random_search = RandomizedSearchCV(RandomForestClassifier(), random_search_params, n_iter=100, cv=5, n_jobs=-1, verbose=1, random_state=42)
        random_search.fit(X_train, y_train)
        best_random_forest = random_search.best_estimator_
        print(f"\nBest parameters for RandomForest (RandomizedSearchCV): {random_search.best_params_}")

        # Evaluate the RandomizedSearchCV model on the test set
        y_pred_random = best_random_forest.predict(X_test)
        print("\nClassification Report for RandomForest (RandomizedSearchCV):")
        print(classification_report(y_test, y_pred_random))