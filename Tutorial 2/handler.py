# This is a little tool to visualise data better than I could simply in c++
import matplotlib.pyplot as mp
import numpy as np
import csv

# Take data into list
with open('data.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

# Turn list into numpy array
array = np.asarray(data)

# Show histogram
mp.hist(array)
mp.show()