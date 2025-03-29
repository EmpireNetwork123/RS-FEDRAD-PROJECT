import flwr as fl
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
import csv
import numpy as np
from model_definition import create_cnn_lstm_model

# Load RS-FEDRAD preprocessed client data
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

# Integrate Basic Iterative Method (BIM) for generating adversarial examples
def generate_adversarial_examples_bim(model, X, y_true, epsilon=0.01, alpha=0.005, iterations=10):
    X_adv = tf.convert_to_tensor(X, dtype=tf.float32)
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    
    # Reshape y_true to match y_pred
    y_true = tf.reshape(y_true, (-1, 1))

    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(X_adv)
            y_pred = model(X_adv)
            loss = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)
        gradients = tape.gradient(loss, X_adv)
        X_adv = X_adv + alpha * tf.sign(gradients)
        X_adv = tf.clip_by_value(X_adv, clip_value_min=0, clip_value_max=1)  # Keep values valid

    return X_adv

# Define the Flower client
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

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Use CSVLogger to save training history per client
        csv_logger = CSVLogger(f'training_history_{self.client_name}.csv', append=True)

        # Generate adversarial examples for training using BIM
        X_train_adv = generate_adversarial_examples_bim(self.model, self.X_train, self.y_train)

        # Combine original and adversarial data
        X_train_combined = np.concatenate([self.X_train, X_train_adv], axis=0)
        y_train_combined = np.concatenate([self.y_train, self.y_train], axis=0)

        # Train the model with combined data (original + adversarial)
        history = self.model.fit(X_train_combined, y_train_combined, epochs=10, batch_size=64, verbose=1, callbacks=[csv_logger])

        # Save metrics to CSV
        with open(f'client_{self.client_name}_metrics.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # Write headers if file is empty
                writer.writerow(['Round', 'Accuracy', 'Loss', 'Precision', 'Recall', 'F1-score'])

            # Generate predictions on the test set
            y_pred_prob = self.model.predict(self.X_test, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype("int32")
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            # Record training metrics
            writer.writerow([config.get('rnd', 'unknown'), history.history['accuracy'][-1], history.history['loss'][-1], precision, recall, f1])

        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Generate adversarial examples for testing using BIM
        X_test_adv = generate_adversarial_examples_bim(self.model, self.X_test, self.y_test)

        # Evaluate on original and adversarial examples
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        loss_adv, accuracy_adv = self.model.evaluate(X_test_adv, self.y_test, verbose=0)

        # Generate predictions
        y_pred_prob = self.model.predict(self.X_test, verbose=0)
        y_pred_adv_prob = self.model.predict(X_test_adv, verbose=0)

        y_pred = (y_pred_prob > 0.5).astype("int32")
        y_pred_adv = (y_pred_adv_prob > 0.5).astype("int32")

        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        precision_adv = precision_score(self.y_test, y_pred_adv)
        recall_adv = recall_score(self.y_test, y_pred_adv)
        f1_adv = f1_score(self.y_test, y_pred_adv)

        return loss, len(self.X_test), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy_adv": accuracy_adv,
            "precision_adv": precision_adv,
            "recall_adv": recall_adv,
            "f1_score_adv": f1_adv
        }

# Run the Flower client
if __name__ == "__main__":
    client_file = "C:/Users/Ericmanny/Desktop/FED_MOdel/ClientData/Cleaned_Company_1_data.csv"
    X_train, X_test, y_train, y_test = load_and_preprocess(client_file)
    model = create_cnn_lstm_model((X_train.shape[1], 1))

    client = RansomwareClient(model, X_train, X_test, y_train, y_test, client_name="Company_1")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)