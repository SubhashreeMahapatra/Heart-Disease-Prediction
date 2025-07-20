import os
import sys
import pickle
#from tkinter import _test
from sqlalchemy import create_engine
from dataclasses import dataclass
import pandas as pd
from exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from logger import logging

@dataclass
class ConnectDBConfig():
    host = 'localhost'
    user = 'root'
    password = 'Sandip@1992'
    database = 'heart'
    table_name = 'heart_disease_data'
    dataset_path:str = os.path.join('dataset', 'heart_disease.csv')


class ConnectDB():
    def __init__(self):
        self.connect_db_config = ConnectDBConfig()

    def retrieve_data(self):
     from sqlalchemy.engine import URL
     url = URL.create(
        drivername="mysql+mysqlconnector",
        username=self.connect_db_config.user,
        password=self.connect_db_config.password,
        host=self.connect_db_config.host,
        port=3306,
        database=self.connect_db_config.database,
    )
     engine = create_engine(url)
     query = f"SELECT * FROM {self.connect_db_config.table_name}"
     df = pd.read_sql(query, engine)

     os.makedirs(os.path.dirname(self.connect_db_config.dataset_path), exist_ok=True)
     df.to_csv(self.connect_db_config.dataset_path, index=False)
     logging.info("Copy of dataset stored in dataset folder as a csv file")
     return df
    


def save_function(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def model_performance(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            report[model_name] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            }
        return report

    except Exception as e:
        raise CustomException(e,sys)
    


def load_obj(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('Error in load_object function in utils')
        raise CustomException(e,sys)
