# Apriori Learning

# To run code in PyCharm console, highlight and press ⌥⇧E (option shift E)

# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Association Rule Learning/Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, 7501):
	transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# transactions.append([item for item in transaction])

# Training
from apyori import apriori
rules = apriori(transactions=transactions, min_support=3 * 7 / 7501, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# Visualing and selecting one with highest lift
results = list(rules)

def inspect(results):
	lhs = [tuple(result[2][0][0])[0] for result in results]
	rhs = [tuple(result[2][0][1])[0] for result in results]
	supports = [result[1] for result in results]
	confidences = [result[2][0][2] for result in results]
	lifts = [result[2][0][3] for result in results]
	return list(zip(lhs, rhs, supports, confidences, lifts))

results_data_frame = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidences', 'Lifts'])

