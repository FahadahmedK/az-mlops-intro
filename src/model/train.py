import argparse
import glob
import os
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import joblib

import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")

    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")

    df = pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

    return df


def split_data(df):
    X = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI',
            'DiabetesPedigree', 'Age']].values
    y = df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data


def train_model(data, args):
    reg_model = LogisticRegression(C=1 / args.reg_rate, solver="liblinear")
    reg_model.fit(data["train"]["X"], data["train"]["y"])
    return reg_model


def get_model_metrics(reg_model, data):

    y_hat = reg_model.predict(data["test"]["X"])
    y_scores = reg_model.predict_proba(data["test"]["X"])
    auc = roc_auc_score(data["test"]["y"], y_scores[:, 1])
    acc = np.average(y_hat == data["test"]["y"])
    metrics = {"accuracy": acc, "auc": auc}
    return metrics


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


def main(args):

    mlflow.autolog()
    # read data
    data = get_csvs_df(args.training_data)

    # Split Data into Training and Validation Sets
    data_dict = split_data(data)
    # Train Model on Training Set

    reg = train_model(data_dict, args)

    # Validate Model on Validation Set
    metrics = get_model_metrics(reg, data_dict)

    logger.info(f"Model metrics: {metrics}")

    # Save Model
    model_name = "sklearn_regression_model.pkl"

    joblib.dump(value=reg, filename=model_name)

    logger.info("Model saved successfully.")


if __name__ == '__main__':
    print("\n\n")
    print("*" * 60)

    mlflow.set_experiment("mlops-experiment")
    args = parse_args()
    with mlflow.start_run():
        main(args)

    print("*" * 60)
    print("\n\n")