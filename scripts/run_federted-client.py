
import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
import csv

# Import model definition from the model_definition file
from model_definition import create_cnn_lstm_model

# Load and preprocess RS-FEDRAD client data
def load_and_preprocess(client_file):
    df = pd.read_csv(client_file)
    X = df.drop(columns=['label'])
    y = df['label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split to balance label distribution in training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)
    return X_train, X_test, y_train, y_test

# Define the Flower client with an extended training process
class RansomwareClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test, client_name):
        self.model = model
        self.X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        self.X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        self.y_train = y_train
        self.y_test = y_test
        self.client_name = client_name

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # CSVLogger to save training history per client
        csv_logger = CSVLogger(f'training_history_{self.client_name}.csv', append=True)

        # Train the model for 10 epochs per round
        history = self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=64, verbose=1, callbacks=[csv_logger])

        # Extract accuracy and loss from the last epoch
        training_metrics = {
            'accuracy': history.history['accuracy'][-1],
            'loss': history.history['loss'][-1]
        }

        # Save round-specific metrics
        with open(f'client_{self.client_name}_metrics.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # Write headers if file is empty
                writer.writerow(['Round', 'Accuracy', 'Loss', 'Precision', 'Recall', 'F1-score'])
            y_pred_prob = self.model.predict(self.X_test, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype("int32")
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            # Use round from config if available
            round_number = config.get('rnd', 'unknown')
            writer.writerow([round_number, training_metrics['accuracy'], training_metrics['loss'], precision, recall, f1])

        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)

        y_pred_prob = self.model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype("int32")

        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        return loss, len(self.X_test), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

# Run the Flower client
if __name__ == "__main__":
    client_file = "C:/Users/Ericmanny/Desktop/FED_MOdel/ClientData/Cleaned_Company_1_data.csv"
    X_train, X_test, y_train, y_test = load_and_preprocess(client_file)

    model = create_cnn_lstm_model((530, 1))
    client = RansomwareClient(model, X_train, X_test, y_train, y_test, client_name="Company_1")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
