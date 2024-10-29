import zipfile
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Function to create sliding windows
def create_windows(data, window_size, step_size):
    num_windows = (len(data) - window_size) // step_size + 1
    windows = []
    
    for i in range(0, num_windows * step_size, step_size):
        window = data.iloc[i:i + window_size]
        windows.append(window)
    
    return windows

# Function to extract features from each window
def extract_window_features(window):
    features = {}
    
    # Compute statistics for each acceleration axis and orientation
    features['accX_mean'] = window['AccelerationX'].mean()
    features['accY_mean'] = window['AccelerationY'].mean()
    features['accZ_mean'] = window['AccelerationZ'].mean()
    
    features['accX_std'] = window['AccelerationX'].std()
    features['accY_std'] = window['AccelerationY'].std()
    features['accZ_std'] = window['AccelerationZ'].std()
    
    features['accX_min'] = window['AccelerationX'].min()
    features['accY_min'] = window['AccelerationY'].min()
    features['accZ_min'] = window['AccelerationZ'].min()
    
    features['accX_max'] = window['AccelerationX'].max()
    features['accY_max'] = window['AccelerationY'].max()
    features['accZ_max'] = window['AccelerationZ'].max()
    
    # Device orientation most frequent value
    features['orientation_mode'] = window['DeviceOrientation'].mode()[0]
    
    return features

# Step 1: Extract the ZIP file
zip_file_path = r'C:\Users\User\Downloads\archive.zip'  # Replace with your actual file path
extraction_dir = 'extracted_files'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

# Step 2: Define the movement type folders
folders = ['downSit', 'freeFall', 'runFall', 'runSit', 'walkFall', 'walkSit']
all_data = []

# Step 3: Loop through each folder and read CSV files
for folder in folders:
    folder_path = os.path.join(extraction_dir, folder)
    label = folder  # Use folder name as the label
    csv_files = os.listdir(folder_path)
    
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        # Read the CSV file
        data = pd.read_csv(csv_path, delimiter=';')
        # Assign the label to all rows in this file based on the folder name
        data['Label'] = label
        # Append the data to the list
        all_data.append(data)

# Combine all the data into a single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Step 4: Data Preprocessing
combined_data = combined_data.dropna()

# Encode DeviceOrientation as numerical
orientation_encoder = LabelEncoder()
combined_data['DeviceOrientation'] = orientation_encoder.fit_transform(combined_data['DeviceOrientation'])

# Step 5: Apply sliding window technique
window_size = 50  # Define window size (e.g., 50 rows per window)
step_size = 25    # Define step size (e.g., move by 25 rows each time)

windows = create_windows(combined_data, window_size, step_size)

# Step 6: Feature extraction for each window
windowed_features = []
window_labels = []

for window in windows:
    # Extract features from each window
    features = extract_window_features(window)
    windowed_features.append(features)
    
    # Assign label to the window based on the majority label within the window
    label = window['Label'].mode()[0]  # Use the most frequent label in the window
    window_labels.append(label)

# Convert the list of windowed features to a DataFrame
X = pd.DataFrame(windowed_features)
y = np.array(window_labels)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 9: Make Predictions
y_pred = clf.predict(X_test)

# Step 10: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_test, y_pred))
