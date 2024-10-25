#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:00:06 2024

@author: Tejinder Pannu
"""
import pandas as pd
import matplotlib.pyplot as plt

# Set the base path where the datasets are located
base_path = "/Users/user/Desktop/SESN HACKATHON/raw-data/"

df_unsup = pd.read_csv(base_path+'output.csv')

df_unsup.head(5)

df_unsup['active_ingredient_name'].nunique()

# Step 1: Select the first 1000 instances
df = df_unsup.head(1000)
df = df_unsup

# Step 2: Drop unnecessary columns
df_cleaned = df.drop(['report_id', 'reaction_id','active_ingredient_id'], axis=1, errors='ignore')
print(df_cleaned.columns)

# Step 3: Rename 'pt_name_eng' to 'adv_effects' and 'soc_name_eng' to 'effect_category'
df_cleaned = df_cleaned.rename(columns={
    'pt_name_eng': 'adv_effects',
    'soc_name_eng': 'effect_category'
})

# Step 4: Arrange the columns in the specified order
column_order = [
    'report_no', 'age', 'weight', 'weight_unit_eng', 
    'gender_eng', 'adv_effects', 'effect_category', 'active_ingredient_name', 'drugname'
]

df_cleaned = df_cleaned[column_order]

# Display the rearranged dataframe and its columns
print(df_cleaned.head())
print(df_cleaned.columns)

#=============================================================================#
#===============================   EDA   =====================================#
#=============================================================================#

# Step 5: EDA - Find the most common drugs and group drugs by adverse effects
top_10_drugs = df_cleaned['drugname'].value_counts().head(10)
drugs_by_effect = df_cleaned.groupby('adv_effects')['drugname'].nunique().reset_index()
drugs_by_effect.columns = ['adv_effects', 'unique_drugs_count']
top_effects = drugs_by_effect.sort_values('unique_drugs_count', ascending=False).head(10)

# Visualization - Top 10 Most Common Drugs
plt.figure(figsize=(10, 5))
plt.barh(top_10_drugs.index, top_10_drugs.values, color='skyblue')
plt.xlabel('Frequency')
plt.ylabel('Drug Name')
plt.title('Top 10 Most Common Drugs')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest frequency on top
plt.show()

# Visualization - Top 10 Adverse Effects by Unique Drug Count
plt.figure(figsize=(12, 6))
plt.barh(top_effects['adv_effects'], top_effects['unique_drugs_count'], color='lightcoral')
plt.xlabel('Number of Unique Drugs')
plt.ylabel('Adverse Effect')
plt.title('Top 10 Adverse Effects by Unique Drug Count')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()

#Plot 3
# Group by 'adv_effects' and 'active_ingredient_name' to count occurrences
ingredients_effect_count = df_cleaned.groupby(['adv_effects', 'active_ingredient_name']).size().reset_index(name='count')

# Find the top 10 ingredients per effect
top_ingredients_per_effect = ingredients_effect_count.sort_values(['adv_effects', 'count'], ascending=[True, False]).groupby('adv_effects').head(10)

# Plotting one bar chart per adverse effect
unique_effects = top_ingredients_per_effect['adv_effects'].unique()

for effect in unique_effects:
    effect_data = top_ingredients_per_effect[top_ingredients_per_effect['adv_effects'] == effect]

    plt.figure(figsize=(12, 6))
    plt.barh(effect_data['active_ingredient_name'], effect_data['count'], color='purple')
    plt.xlabel('Occurrence Count')
    plt.ylabel('Active Ingredient Name')
    plt.title(f'Top Ingredients for Adverse Effect: {effect}')
    plt.gca().invert_yaxis()
    plt.show()
    
df_unsup = df_cleaned[['adv_effects', 'active_ingredient_name']]

#=============================================================================#
#===========================   Unsupervised   ================================#
#=============================================================================#

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Use separate encoders to avoid conflicts in feature names
ingredient_encoder = OneHotEncoder(sparse_output=False)
effect_encoder = OneHotEncoder(sparse_output=False)

# Encode 'active_ingredient_name'
ingredient_matrix = ingredient_encoder.fit_transform(df_cleaned[['active_ingredient_name']])
df_ingredient = pd.DataFrame(ingredient_matrix, 
                             columns=ingredient_encoder.get_feature_names_out())
# Encode 'adv_effects'
effect_matrix = effect_encoder.fit_transform(df_cleaned[['adv_effects']])
df_effects = pd.DataFrame(effect_matrix, 
                          columns=effect_encoder.get_feature_names_out())

# Combine the two encoded matrices
df_cluster = pd.concat([df_ingredient, df_effects], axis=1)

# Verify the combined DataFrame
print(df_cluster.head())

# Apply KMeans clustering
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_cluster)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve to determine the best k
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal k')
plt.show()

# Use the optimal k to fit the model
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['cluster'] = kmeans.fit_predict(df_cluster)

# Display the cluster assignments with original features for analysis
df_result = pd.concat([df_cleaned[['active_ingredient_name', 'adv_effects']], 
                       df_cluster['cluster']], axis=1)
print(df_result.head())

#=============================================================================#
#===========================   Deep Learning   ===============================#
#=============================================================================#
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Label Encode 'active_ingredient_name' and 'adv_effects'
ingredient_encoder = LabelEncoder()
effect_encoder = LabelEncoder()

df_cleaned['ingredient_label'] = ingredient_encoder.fit_transform(df_cleaned['active_ingredient_name'])
df_cleaned['effect_label'] = effect_encoder.fit_transform(df_cleaned['adv_effects'])

# Step 2: Scale the Data
scaler = StandardScaler()

# Combine the features to scale them together
X = np.column_stack((df_cleaned['ingredient_label'], df_cleaned['effect_label']))
X_scaled = scaler.fit_transform(X)

# Use the effect labels as target
y = df_cleaned['effect_label'].values

# Step 3: Split the Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Unpack the scaled features
X_train_ingredients, X_train_effects = X_train[:, 0], X_train[:, 1]
X_test_ingredients, X_test_effects = X_test[:, 0], X_test[:, 1]

print(f"Training set size: {len(y_train)}, Test set size: {len(y_test)}")

# Step 4: Create the Embedding Model
num_ingredients = len(ingredient_encoder.classes_)
num_effects = len(effect_encoder.classes_)

# Ingredient embedding
ingredient_input = Input(shape=(1,), name='ingredient_input')
ingredient_embedding = Embedding(input_dim=num_ingredients, output_dim=32, name='ingredient_embedding')(ingredient_input)
ingredient_flatten = Flatten()(ingredient_embedding)

# Effect embedding
effect_input = Input(shape=(1,), name='effect_input')
effect_embedding = Embedding(input_dim=num_effects, output_dim=32, name='effect_embedding')(effect_input)
effect_flatten = Flatten()(effect_embedding)

# Concatenate embeddings and pass through dense layers
concatenated = Concatenate()([ingredient_flatten, effect_flatten])
dense_1 = Dense(64, activation='relu')(concatenated)
dense_2 = Dense(32, activation='relu')(dense_1)
output = Dense(num_effects, activation='softmax', name='output')(dense_2)

# Step 5: Create and Compile the Model
model = Model(inputs=[ingredient_input, effect_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 6: Apply Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Step 7: Train the Model with Early Stopping
history = model.fit(
    [X_train_ingredients, X_train_effects], y_train,
    validation_data=([X_test_ingredients, X_test_effects], y_test),
    epochs=50,  # Increased epoch count to leverage early stopping
    batch_size=32,
    callbacks=[early_stopping]
)

# Step 8: Plot Training and Validation Metrics
# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Step 9: Evaluate the Model on Test Data
test_loss, test_accuracy = model.evaluate([X_test_ingredients, X_test_effects], y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Step 10: Make Predictions
y_pred = model.predict([X_test_ingredients, X_test_effects])
y_pred_classes = np.argmax(y_pred, axis=1)

# Compare predictions with true labels
print("Sample predictions:", y_pred_classes[:5])
print("Actual labels:", y_test[:5])

# Step 11: Classification Report and Confusion Matrix
class_names = effect_encoder.inverse_transform(np.unique(y_test))

print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
