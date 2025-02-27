import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from pycaret.regression import setup
import os

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Load dataset
    # Manually set the working directory to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    os.chdir(project_root)

    dataset_path = cfg.dataset.raw_path
    abs_path = os.path.abspath(dataset_path)  # Convert relative path to absolute path

    df = pd.read_csv(abs_path)

    # Drop initial unnecessary columns
    df_cleaning = df.drop(columns=cfg.dataset.drop_columns_initial)

    # Fill missing values
    df_cleaning["BuildingArea"] = df_cleaning["BuildingArea"].fillna(df_cleaning["BuildingArea"].median())
    df_cleaning["Car"] = df_cleaning["Car"].fillna(df_cleaning["Car"].median())
    df_cleaning["YearBuilt"] = df_cleaning["YearBuilt"].fillna(df_cleaning["YearBuilt"].median())

    # Calculate Age
    df_cleaning["Date"] = pd.to_datetime(df_cleaning["Date"], format="%d/%m/%Y")
    df_cleaning["Sale Year"] = df_cleaning["Date"].dt.year
    df_cleaning["Age"] = df_cleaning["Sale Year"] - df_cleaning["YearBuilt"]
    df_cleaning["Age"] = df_cleaning["Age"].astype(int)

    # Drop final columns after calculations
    df_cleaning = df_cleaning.drop(columns=cfg.dataset.drop_columns_final)

    # Split the data
    train_data, test_data = train_test_split(df_cleaning,
                                             test_size=cfg.train_test_split.test_size,
                                             random_state=cfg.train_test_split.random_state)

    # Save processed data
    train_data.to_csv(cfg.dataset.processed_train_path, index=False)
    test_data.to_csv(cfg.dataset.processed_test_path, index=False)
    print(f"Preprocessing complete. Train set saved to {cfg.dataset.processed_train_path}, Test set saved to {cfg.dataset.processed_test_path}")

    print('checkpoint')
    # Initialize PyCaret
    # setup(data=train_data,
    #       target=cfg.pycaret_setup.target,
    #       session_id=cfg.pycaret_setup.session_id,
    #       normalize=cfg.pycaret_setup.normalize,
    #       transformation=cfg.pycaret_setup.transformation,
    #       remove_outliers=cfg.pycaret_setup.remove_outliers,
    #       remove_multicollinearity=cfg.pycaret_setup.remove_multicollinearity,
    #       multicollinearity_threshold=cfg.pycaret_setup.multicollinearity_threshold
    #       )

if __name__ == "__main__":
    main()
