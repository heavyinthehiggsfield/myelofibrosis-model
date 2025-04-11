# Myelofibrosis: Modeling Patient Time-to-Event and Survivalability Classification

## Run Instructions
1. git clone the repo

2. create new venv (virtual environment) and select it as your python interpreter (in your IDE)


3. source your new venv (aka code will from this environment)
    source venv/bin/activate

4. Install requirements.txt (if you still have trouble with imports you can pip install each one)
    pip install -r requirements.txt

5. Run syntheticData.py script to validate python setup

6. Run trainTest.py to validate XGBoost setup (set to randomForest until below is set)
    If issues appear regarding libomp, you may need to install globally on your local:
        brew install libomp
    which may first require MacOS Command line tools:
        xcode-select --install

If successful, this should mean that all major functionality is working as expected - real data can be pointed to in place of the synthetic data and training can begin, with testing/optimizing/tracking done simultaneously using Optuna & Weights and Biases (setup for which will be included in next commit)

Note: any model trained within this project will be saved in the '/models' folder
To Run a prediction, use predict.py and specify the model?

# 🧱 Overall Pipeline
- Load the data

- Preprocess (handle categorical, normalize/scale if needed)

- Train-test split

- Train XGBoost regressor

- Evaluate with relevant metrics

- (Optional) Hyperparameter tuning with Optuna

- (Optional) Log everything with Weights & Biases

## Some Major Project Components
1. preprocess.py
    data setup before training. This will likely contain a generic script agnostic to the data we want to inlude (which will be included in the main train.py script)

2. config.py
    main configuration zone, environment variables / specifics / etc. that all scripts can be set to pull from so all 'configs' are localized to one file (and can be verified as such before running)

3. testTrain.py
    just a generic XGBoost train script without cross validation etc.

4. train.py
    Main starting point for training based on info set in config.py

5. Stripped down components, good for separating MLOps flows / notebook integrations:
- dataset.py 
    (main data setup pre-training)
- train.py 
    (main training hub)
- predict.py 
    (barebones prediction)
- evaluate.py 
    (testing / evaluating pipeline)
- optuna.py 
    (hyperparameter tuning / optimization)
    

## Testing Plan ?

## Anything else ?
