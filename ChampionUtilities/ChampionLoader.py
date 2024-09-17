import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load the CSV data into a pandas DataFrame
csv_file_path = '../Data/LoL-Champions.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)

# Assuming the first column is the champion names
champion_names = df.iloc[:, 0]  # First column with champion names

# Identify numeric and non-numeric columns
numeric_features = df.select_dtypes(include=['float64', 'int64']).copy()
non_numeric_features = df.select_dtypes(exclude=['float64', 'int64']).copy()

# Remove the first column (champion names) from non-numeric features
non_numeric_features = non_numeric_features.iloc[:, 1:]

# Encode non-numeric features using one-hot encoding
encoder = OneHotEncoder()
encoded_non_numeric = encoder.fit_transform(non_numeric_features)

# Convert numeric features to a NumPy array
numeric_features_array = numeric_features.values

print(encoded_non_numeric)
print(numeric_features_array)

# Concatenate numeric and encoded non-numeric features
all_features_array = np.concatenate((numeric_features_array, encoded_non_numeric), axis=1)

# Convert the concatenated array to a tensor
features_tensor = torch.tensor(all_features_array, dtype=torch.float32)

# Optionally, keep a list of the champion names for reference
champion_list = champion_names.tolist()

# Print the shape of the tensor and some example data
print("Shape of the tensor:", features_tensor.shape)
print("Example data from tensor:", features_tensor[:5])
