MLOPS assignment - Development and deployment

--Overview--
This repository contains the implementation of an end-to-end MLOps pipeline for training and deploying machine learning models. This assignment will follow the best practices in model productization which includes 
- Standard ML project folder structure
- Poetry for dependency management
- Hydra for configuration management
- DVC for data version control
- Collaborative work through GitHub
- Flask-based web application deployment

--Project structure--

The main file would be mlops-assignment and there will be 4 folders , xinrui , dekai , jet , common.
In each of the folder for xinrui , dekai and jet - There will be folders for data folder, notebooks which contains task 1 and task 2 jupyter notebook , src folder as well as app folder for task 3 . In the app folder, there will be app.py , saved ML model pkl file and there would be templates folder for the index.html.

Dekai will be doing melboune residential dataset , jet will be using used car prices and xinrui will be doing the wheat seeds dataset.

For the common folder , there is pyproject.toml and this README.md file.

--Development and deployment workflow--
1. Data preparation and preprocessing
- Raw data is stored in data/raw/
- Processed data versioned using DVC

2. Model training and evaluation
- Models are trained using PyCaret and managed via Hydra configurations
- Hyperparameter tuning is performed for better accuracy

3. Model Deployment
- A Flask-based web application is developed to serve predictions
- Docker containerization for cloud deployment

4. Version control and collaboration
- Dependencies managed using Poetry
- Code collaboration via GitHub

The app folder in the main directory is a centralized app.py to serve all the 3 models in a single deployment.

-- Summarized-- Each person's folder has

Src ( Contains train.py , preprocess.py , config.yaml)
models (This pkl file will be developed after running the train.py)
plot (Optional)
processed (The train and test datasets will be developed after running preprocess.py)
notebooks (Previous individual task 1 and task 2 jupyter notebooks. Individual files!)
data (Inside theres foolder called raw , inside theres individual given csv/excel)
app (This is individual task 3 folder. Should be able to run with poetry. ) Folders
