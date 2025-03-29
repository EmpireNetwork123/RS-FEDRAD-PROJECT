
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#  First we Load our 4 datasets
data_dir = r"C:\Users\Ericmanny\Desktop\FED_MOdel\DataSet"

ransomware_1 = pd.read_csv(os.path.join(data_dir, "Ramsomware_1.csv"), low_memory=False).values
ransomware_2 = pd.read_csv(os.path.join(data_dir, "Ramsomware_2.csv"), low_memory=False).values
ransomware_3 = pd.read_csv(os.path.join(data_dir, "Ramsomware_3.csv"), low_memory=False)
benign = pd.read_csv(os.path.join(data_dir, "Benign.csv"), low_memory=False)

# Convert all columns to numeric, forcing errors to NaN
ransomware_3 = ransomware_3.apply(pd.to_numeric, errors='coerce').values
benign = benign.apply(pd.to_numeric, errors='coerce').values

#  Create labels
ransomware_labels_1 = np.ones(len(ransomware_1))
ransomware_labels_2 = np.ones(len(ransomware_2))
ransomware_labels_3 = np.ones(len(ransomware_3))
benign_labels = np.zeros(len(benign))

# Combine all our datasets
X_combined = np.concatenate((ransomware_1, ransomware_2, ransomware_3, benign))
y_combined = np.concatenate((ransomware_labels_1, ransomware_labels_2, ransomware_labels_3, benign_labels))

# We Standardize the dataset
scaler = StandardScaler()
X_combined = scaler.fit_transform(X_combined)

# Step 6: Shuffle the combined dataset
X_combined, y_combined = shuffle(X_combined, y_combined, random_state=42)

# Step 7: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Carry out Proportional split into 10 clients
num_clients = 10
ransomware_1_split = np.array_split(ransomware_1[:len(X_train)], num_clients)
ransomware_2_split = np.array_split(ransomware_2[:len(X_train)], num_clients)
ransomware_3_split = np.array_split(ransomware_3[:len(X_train)], num_clients)
benign_split = np.array_split(benign[:len(X_train)], num_clients)

clients_data = []
for i in range(num_clients):
    X_client = np.concatenate((ransomware_1_split[i], ransomware_2_split[i], ransomware_3_split[i], benign_split[i]))
    y_client = np.concatenate((
        np.ones(len(ransomware_1_split[i])),  # Ransomware 1 labels (1)
        np.ones(len(ransomware_2_split[i])),  # Ransomware 2 labels (1)
        np.ones(len(ransomware_3_split[i])),  # Ransomware 3 labels (1)
        np.zeros(len(benign_split[i]))  # Benign labels (0)
    ))
    # Shuffle client data
    X_client, y_client = shuffle(X_client, y_client, random_state=42)
    clients_data.append((X_client, y_client))

    # Here we save each client's data with the desired naming convention
    np.save(os.path.join(data_dir, f"Cleaned_Company_{i+1}_data_X.npy"), X_client)
    np.save(os.path.join(data_dir, f"Cleaned_Company_{i+1}_data_y.npy"), y_client)

# Importantly we save training and testing data
np.save(os.path.join(data_dir, "X_train.npy"), X_train)
np.save(os.path.join(data_dir, "y_train.npy"), y_train)
np.save(os.path.join(data_dir, "X_test.npy"), X_test)
np.save(os.path.join(data_dir, "y_test.npy"), y_test)

print("Data processing and saving completed!")
