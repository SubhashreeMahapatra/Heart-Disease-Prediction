import os
import sys 

# Add the parent directory of "source" to sys.path if not already present
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logger import logging
from exception import CustomException
from utils import ConnectDB
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Step 1: Create path variables to store the files as raw csv
@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('datalog', 'train.csv')
    test_data_path:str = os.path.join('datalog', 'test.csv')
    raw_data_path:str = os.path.join('datalog', 'raw.csv')


# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method started')
        try:
            connect_db = ConnectDB()
            connect_db.retrieve_data()
            df = pd.read_csv(os.path.join('dataset', 'heart_disease.csv'))
            logging.info('Dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Train Test split')
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is complete')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        

        except Exception as e:
            logging.info('Exception occured at Data Ingestion Stage')
            raise CustomException(e,sys)