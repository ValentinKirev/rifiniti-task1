import pandas as pd
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
