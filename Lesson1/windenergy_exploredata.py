#Explore the Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#funtion to import data as a pandas dataframe

def datareader(csvfile):
    data = pd.read_csv(csvfile)
    return data

#import data
winddata = datareader("windDatahr_month.csv")

#return info of the dataframe
winddata.info()

#return the first n rows
winddata.head(n=2)

#generate descriptive statistics
winddata.describe(include="all")

#compute pairwise correlation of columns
corr_matrix = winddata.corr()
corr_matrix["Energy_kWh"]

# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["Time", "Speed", "Direction","Energy_kWh"]
scatter_matrix(winddata[attributes], figsize=(12, 8))
plt.show()

winddata.plot(kind="scatter", x="Speed", y="Energy_kWh")
plt.show()
