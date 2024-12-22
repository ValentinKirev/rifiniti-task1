import pandas as pd


# load data as Pandas DataFrame
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

data_frame = pd.read_csv(url)

# Print Summary statistics for each feature
print("Summary statistics: ")
print(data_frame.describe())
