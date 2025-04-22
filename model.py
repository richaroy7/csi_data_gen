import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the first 200 data points
df_first = pd.read_csv('first_200_combined_data.csv', encoding='utf-8')

# Load the last 50 data points
df_last = pd.read_csv('last_50_combined_data.csv', encoding='utf-8')

# Combine the two DataFrames into a single training dataset
df = pd.concat([df_first, df_last], ignore_index=True)

print(f"Total number of data points for training: {len(df)}")
