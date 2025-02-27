import os
import pandas as pd
import hydra
import matplotlib
import shutil
import logging  # Import logging module

matplotlib.use("Agg") 
from omegaconf import DictConfig
from pycaret.classification import (
    setup, compare_models, evaluate_model, plot_model, 
    predict_model, create_model, save_model
)

# Initialize logging to write logs into the same directory as train.py
log_dir = os.path.dirname(os.path.abspath(__file__))  # Get train.py directory
log_file = os.path.join(log_dir, "training_logs.txt")

# Configure logging
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

@hydra.main(config_path=".", config_name="config", version_base=None)
def train_model(cfg: DictConfig):
    # Ensure working directory is set to where train.py is located
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    dataset_path = os.path.abspath(cfg.dataset.processed_train_path)
    logging.info(f"Checking dataset path: {dataset_path}")
    print(f"Checking dataset path: {dataset_path}")  # Print for terminal view

    if not os.path.exists(dataset_path):
        logging.error(f"Processed dataset file '{dataset_path}' not found. Run preprocess.py first.")
        raise FileNotFoundError(f"Processed dataset file '{dataset_path}' not found. Run preprocess.py first.")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Set up PyCaret
    clf_setup = setup(
        data=df,
        target="Type",
        normalize=cfg.training.normalize,
        feature_selection=cfg.training.feature_selection,
        bin_numeric_features=cfg.training.get("bin_features", []),  # Avoid error if missing
        session_id=cfg.training.session_id,
        log_experiment=cfg.training.get("log_experiment", False),
    )

    # Automatically select best model or use predefined one
    if cfg.model.auto_select:
        logging.info("Auto-selecting best model...")
        best_model = compare_models(sort="Accuracy")
    else:
        if not cfg.model.type:
            logging.error("Model type must be specified in config.yaml if auto_select is False.")
            raise ValueError("Model type must be specified in config.yaml if auto_select is False.")
        
        logging.info(f"Creating model: {cfg.model.type}")
        best_model = create_model(cfg.model.type)

    # Evaluate the model
    logging.info("Evaluating model performance...")
    evaluate_model(best_model)
    
    save_dir = os.path.join(log_dir, "plots")  # Save plots in train.py directory
    os.makedirs(save_dir, exist_ok=True)

    plot_model(best_model, plot="confusion_matrix", save=True)
    plot_model(best_model, plot="auc", save=True)

    # Move plots to designated directory
    shutil.move("Confusion Matrix.png", f"{save_dir}/confusion_matrix.png")
    shutil.move("AUC.png", f"{save_dir}/auc.png")

    logging.info(f"Confusion Matrix saved to: {save_dir}/confusion_matrix.png")
    logging.info(f"AUC Curve saved to: {save_dir}/auc.png")

    if hasattr(best_model, "coef_") or hasattr(best_model, "feature_importances_"):
        plot_model(best_model, plot="feature")
    else:
        logging.info("Feature Importance plot skipped as model does not support it.")

    # Generate predictions
    logging.info("Making predictions on test data...")
    predictions = predict_model(best_model)

    # Save the trained model
    model_path = os.path.join(log_dir, cfg.model.output_path)  # Save model in train.py directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
    save_model(best_model, model_path)

    logging.info(f"Model saved to {model_path}")

    return predictions

if __name__ == "__main__":
    logging.info("Starting training script...")
    train_model()
    logging.info("Training completed successfully.")
