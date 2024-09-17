import pandas as pd
import random
from sklearn.model_selection import train_test_split







# Example of iterating over training data
print("Training Data:")
for blue_team, red_team, label in train_matches[:5]:  # Just showing 5 for demonstration
    print("Blue Team:", blue_team)
    print("Red Team:", red_team)
    print("Label:", label)
    print("---" * 10)

print("Testing Data:")
for blue_team, red_team, label in test_matches[:5]:  # Just showing 5 for demonstration
    print("Blue Team:", blue_team)
    print("Red Team:", red_team)
    print("Label:", label)
    print("---" * 10)
