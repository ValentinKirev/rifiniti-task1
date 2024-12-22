import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data as Pandas DataFrame
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

data_frame = pd.read_csv(url, names=columns_names)

# Print Summary statistics for each feature
print("Summary statistics: ")
print(f"{data_frame.describe()}\n")

# Print count of unique values in the "species" column
print("Count of unique values in the 'species' column: ")
print(f"{data_frame['species'].value_counts()}\n")

# Create histogram for "Sepal Length" column
plt.hist(data_frame["sepal_length"], bins=25, color="blue", edgecolor="black")
plt.title("Histogram of Sepal Lenght")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

# Create Scatter plot for comparing "Sepal Length" column and "Sepal Width" column
plt.scatter(data_frame["sepal_length"], data_frame["sepal_width"], color="blue")
plt.title("Scatter plot for comparing Sepal Length and Sepal Width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

# Identifies missing values in the dataset
print("Missing values in the dataset: ")
print(f"{data_frame.isnull().sum()}\n")

# Simulating missing values
simulated_df = data_frame.copy()
simulated_df["sepal_length"] = simulated_df["sepal_length"].replace(5.9, None)

print("Result simulated with missing values in 'Sepal length' column: ")
print(f"{simulated_df.isnull().sum()}\n")

# Handling missing values
simulated_df["sepal_length"].fillna(simulated_df["sepal_length"].mean(), inplace=True)

# Verify that missing values are handled
print("Result with handled missing values: ")
print(f"{simulated_df.isnull().sum()}\n")
