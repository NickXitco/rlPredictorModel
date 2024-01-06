import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Generating a mock dataset
np.random.seed(0)  # For reproducibility

# Let's assume we have data for 1000 matches, with 90 time slices each (roughly one per minute)
num_matches = 1000
time_slices_per_match = 90

# Creating random data
data = {
    "time_remaining": np.tile(np.arange(time_slices_per_match, 0, -1) * 60, num_matches),
    "goals_for": np.random.randint(0, 5, num_matches * time_slices_per_match),
    "goals_against": np.random.randint(0, 5, num_matches * time_slices_per_match),
    "shots_on_target_for": np.random.randint(0, 10, num_matches * time_slices_per_match),
    "shots_on_target_against": np.random.randint(0, 10, num_matches * time_slices_per_match),
    "saves_for": np.random.randint(0, 10, num_matches * time_slices_per_match),
    "saves_against": np.random.randint(0, 10, num_matches * time_slices_per_match),
    "possession": np.random.randint(40, 60, num_matches * time_slices_per_match),
    "win": np.random.randint(0, 2, num_matches * time_slices_per_match)
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Displaying the first few rows of the dataset
df.head()

# Step 3: Splitting the dataset
features = df.drop('win', axis=1)
target = df['win']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# Step 4: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generating a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print(accuracy, conf_matrix)


# Selecting a random time slice from the dataset
sample_slice = df.drop('win', axis=1).sample(1)

# Displaying the selected slice
sample_slice_display = sample_slice.copy()
sample_slice_display['win'] = model.predict(sample_slice)  # Adding the predicted win/loss

# Using predict_proba to get the win probabilities for the sample slice
win_probabilities = model.predict_proba(sample_slice)

# Adding the win probability to our display
sample_slice_display['win_probability'] = win_probabilities[:, 1]  # Probability of class 1 (win)

print(sample_slice_display)



