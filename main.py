import pandas as pd


# load data as Pandas DataFrame
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

data_frame = pd.read_csv(url)
