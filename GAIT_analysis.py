import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the IMU dataset
data = pd.read_csv('./imu_data_20240820_160834.csv')
accel_x, accel_y, accel_z = data['Ax'], data['Ay'], data['Az']
gyro_x, gyro_y, gyro_z = data['Gx'], data['Gy'], data['Gz']

# Calculate the magnitude of acceleration and gyroscope data
accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
gyro_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

# Define thresholds based on statistical analysis
accel_mean = accel_magnitude.mean()
accel_std = accel_magnitude.std()
gyro_mean = gyro_magnitude.mean()
gyro_std = gyro_magnitude.std()

# Define thresholds (e.g., 3 standard deviations from mean for anomaly detection)
accel_anomaly_threshold = accel_mean + 3 * accel_std
gyro_anomaly_threshold = gyro_mean + 3 * gyro_std

# Define parameters for fall detection (thresholds based on domain knowledge)
accel_threshold = 12  # Example threshold for high acceleration indicating a fall
gyro_threshold = 50   # Example threshold for high rotation speed indicating a fall
window_size = 50      # Adjust window size based on data sampling rate

# Anomaly detection using sliding window approach
fall_indices = []
for i in range(0, len(data) - window_size, window_size):
    window_accel_magnitude = accel_magnitude[i:i+window_size]
    window_gyro_magnitude = gyro_magnitude[i:i+window_size]
    
    # Check for both anomaly and fall thresholds
    if (window_accel_magnitude.max() > accel_anomaly_threshold or window_accel_magnitude.max() > accel_threshold) \
        and (window_gyro_magnitude.max() > gyro_anomaly_threshold or window_gyro_magnitude.max() > gyro_threshold):
        fall_indices.append(i + np.argmax(window_accel_magnitude))

# Plotting acceleration and gyroscope magnitude with detected anomalies
plt.figure(figsize=(12, 6))

# Plot acceleration magnitude
plt.subplot(2, 1, 1)
plt.plot(accel_magnitude, label='Acceleration Magnitude', color='blue')
plt.scatter(fall_indices, accel_magnitude[fall_indices], color='red', label='Detected Falls')
plt.axhline(accel_anomaly_threshold, color='purple', linestyle='--', label='Anomaly Threshold')
plt.title('Acceleration Magnitude with Detected Falls')
plt.xlabel('Time')
plt.ylabel('Acceleration Magnitude')
plt.legend()

# Plot gyroscope magnitude
plt.subplot(2, 1, 2)
plt.plot(gyro_magnitude, label='Gyroscope Magnitude', color='orange')
plt.scatter(fall_indices, gyro_magnitude[fall_indices], color='red', label='Detected Falls')
plt.axhline(gyro_anomaly_threshold, color='purple', linestyle='--', label='Anomaly Threshold')
plt.title('Gyroscope Magnitude with Detected Falls')
plt.xlabel('Time')
plt.ylabel('Gyroscope Magnitude')
plt.legend()

plt.tight_layout()
plt.show()

# Output detected fall points with timestamps
fall_times = data['Timestamp'].iloc[fall_indices]
print("Timestamps of detected falls:", fall_times.values)
print(f"Total falls detected: {len(fall_indices)}")