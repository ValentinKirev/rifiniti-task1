import pandas as pd

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
