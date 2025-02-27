import os
import pandas as pd
import hydra
import shutil
import logging
from omegaconf import DictConfig
from pycaret.regression import (
    setup, compare_models, evaluate_model, plot_model,
    predict_model, create_model, save_model, tune_model
)

# Initialize logging
logging.basicConfig(
    filename="training_logs.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

@hydra.main(config_path=".", config_name="config", version_base=None)
def train_model(cfg: DictConfig):
    # Now that Hydra is initialized, get the content root
    project_root = hydra.utils.get_original_cwd()
    os.chdir(project_root)  # Change the working directory to the content root
    print(project_root)

    log_dir = os.path.join(project_root, "logs")  # Ensure logs are in the project root
    log_file = os.path.join(log_dir, "training_logs.txt")

    # Make sure the logs directory exists
    os.makedirs(log_dir, exist_ok=True)

    train_dataset_path = os.path.join(project_root, "mlops-assignment", cfg.dataset.processed_train_path)
    train_dataset_path = os.path.abspath(train_dataset_path)  # Convert to absolute path

    logging.info(f"Checking dataset path: {train_dataset_path}")
    print(f"Checking dataset path: {train_dataset_path}")

    if not os.path.exists(train_dataset_path):
        logging.error(f"Dataset not found at {train_dataset_path}. Run preprocess.py first.")
        raise FileNotFoundError(f"Dataset '{train_dataset_path}' not found.")

    train_data = pd.read_csv(train_dataset_path)

    # Initialize PyCaret setup
    s = setup(data=train_data,
              target=cfg.pycaret_setup.target,
              session_id=cfg.pycaret_setup.session_id,
              normalize=cfg.pycaret_setup.normalize,
              transformation=cfg.pycaret_setup.transformation,
              remove_outliers=cfg.pycaret_setup.remove_outliers,
              remove_multicollinearity=cfg.pycaret_setup.remove_multicollinearity,
              multicollinearity_threshold=cfg.pycaret_setup.multicollinearity_threshold,
              experiment_name='Dekai_Melbourne_Residential_price_predictions')

    # Auto-select best model
    if cfg.model.auto_select:
        logging.info("Auto-selecting best regression model...")
        best_model = compare_models(sort="RMSE")
    else:
        if not cfg.model.type:
            logging.error("Model type must be specified if auto_select is False.")
            raise ValueError("Model type must be specified in config.yaml")

        logging.info(f"Creating model: {cfg.model.type}")
        best_model = create_model(cfg.model.type)

    # Tune the model to optimize hyperparameters
    logging.info("Tuning model hyperparameters...")
    tuned_model = tune_model(best_model, optimize="RMSE")

    # Evaluate model
    logging.info("Evaluating model performance...")
    evaluate_model(tuned_model)

    # Save regression-specific plots
    save_dir = os.path.join("plots")
    os.makedirs(save_dir, exist_ok=True)

    plot_model(tuned_model, plot="residuals", save=True)
    plot_model(tuned_model, plot="error", save=True)
    plot_model(tuned_model, plot="feature", save=True)

    shutil.move("Residuals.png", f"{save_dir}/residuals.png")
    shutil.move("Prediction Error.png", f"{save_dir}/error.png")
    shutil.move("Feature Importance.png", f"{save_dir}/feature.png")

    logging.info(f"Residuals Plot saved: {save_dir}/residuals.png")
    logging.info(f"Prediction Error Plot saved: {save_dir}/error.png")
    logging.info(f"Feature Importance saved: {save_dir}/feature.png")

    # Save final trained model
    model_path = os.path.join(project_root, "mlops-assignment", cfg.model.output_path + "_without_mlflow.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_model(tuned_model, model_path)
    logging.info(f"Model saved to {model_path}")

    # # saving to app folder
    model_path_app = os.path.join(project_root, "mlops-assignment", cfg.model.output_path_app)
    os.makedirs(os.path.dirname(model_path_app), exist_ok=True)
    save_model(tuned_model, model_path_app)
    #
    # logging.info(f"Model saved to {model_path_app}")

    return tuned_model

if __name__ == "__main__":
    logging.info("Starting training script...")
    train_model()
    logging.info("Training completed successfully.")
