import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from a CSV file into a DataFrame
file_path = 'project_1_Data.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Perform statistical analysis
print(df.describe())

# Visualize the dataset behaviour within each class
# Assuming 'class' is the column name for the classes in the dataset
if 'class' in df.columns:
    # Pairplot to visualize relationships between features within each class
    sns.pairplot(df, hue='class')
    plt.show()

    # Boxplot to visualize the distribution of each feature within each class
    for column in df.columns:
        if column != 'class':
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='class', y=column, data=df)
            plt.title(f'Boxplot of {column} by class')
            plt.show()
else:
    print("The dataset does not contain a 'class' column.")