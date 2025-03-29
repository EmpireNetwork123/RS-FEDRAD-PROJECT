
import flwr as fl
import logging
import csv
from model_definition import create_cnn_lstm_model  # Ensure you have the correct model import

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)

# Function to save the global model after each round
def save_model(global_model, round_number):
    model_path = f"global_model_round_{round_number}.h5"
    global_model.save(model_path)
    logging.info(f"Global model for round {round_number} saved as {model_path}")

# Function to save metrics to CSV after each round
def save_metrics_to_csv(metrics, round_number):
    with open(f'global_metrics_round_{round_number}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if round_number == 1:  # Write headers only for the first round
            writer.writerow(['Round', 'Accuracy', 'Loss', 'Precision', 'Recall', 'F1-score'])
        writer.writerow([round_number, metrics.get('accuracy', 'N/A'), metrics.get('loss', 'N/A'),
                         metrics.get('precision', 'N/A'), metrics.get('recall', 'N/A'), metrics.get('f1_score', 'N/A')])

# Custom FedAvg strategy to aggregate metrics and save the model
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics = []

    def aggregate_fit(self, rnd, results, failures):
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        if aggregated_result is not None:
            aggregated_parameters, _ = aggregated_result

            # Ensure results contain metrics
            metrics_list = [res.metrics for res, _ in results if hasattr(res, 'metrics') and res.metrics]
            
            if metrics_list:
                # Aggregate client metrics
                round_accuracy = sum(m['accuracy'] for m in metrics_list) / len(metrics_list)
                round_loss = sum(m['loss'] for m in metrics_list) / len(metrics_list)
                round_precision = sum(m['precision'] for m in metrics_list) / len(metrics_list)
                round_recall = sum(m['recall'] for m in metrics_list) / len(metrics_list)
                round_f1 = sum(m['f1_score'] for m in metrics_list) / len(metrics_list)

                metrics = {
                    'accuracy': round_accuracy,
                    'loss': round_loss,
                    'precision': round_precision,
                    'recall': round_recall,
                    'f1_score': round_f1
                }
                self.round_metrics.append(metrics)

                # Save metrics and model
                save_metrics_to_csv(metrics, rnd)

                # Save model weights
                final_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
                global_model = create_cnn_lstm_model((530, 1))  # Ensure the input shape is correct for your model
                global_model.set_weights(final_weights)
                save_model(global_model, rnd)

            return aggregated_result
        return None

# Start the Flower server
if __name__ == "__main__":
    strategy = CustomFedAvg()
    logging.info("Starting Flower server...")

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),  # Change num_rounds to 10 for more rounds
        strategy=strategy
    )

    # After training completes, log the final metrics
    logging.info("Training completed.")
    if strategy.round_metrics:
        logging.info(f"Final metrics: Accuracy - {strategy.round_metrics[-1]['accuracy']}")
        logging.info(f"Final metrics: Loss - {strategy.round_metrics[-1]['loss']}")
        logging.info(f"Final metrics: Precision - {strategy.round_metrics[-1]['precision']}")
        logging.info(f"Final metrics: Recall - {strategy.round_metrics[-1]['recall']}")
        logging.info(f"Final metrics: F1-score - {strategy.round_metrics[-1]['f1_score']}")
