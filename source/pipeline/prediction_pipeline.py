import sys
import os
# Add the parent directory of "srouce" to sys.path if not already present
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from exception import CustomException
from logger import logging
from utils import load_obj
import pandas as pd

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('datalog', 'preprocessor.pkl')
            model_path = os.path.join('datalog', 'model.pkl')

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info('Error occured in predict function in prediction pipleine')
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self, age:float,
                 systolic_bp:float,
                 cholesterol:float,
                 gender:str,
                 diabetes:str,
                 smoking_status:str,
                 chest_pain_type:str):
        self.age = age
        self.systolic_bp = systolic_bp
        self.cholesterol = cholesterol
        self.gender = gender
        self.diabetes = diabetes
        self.smoking_status = smoking_status
        self.chest_pain_type = chest_pain_type

    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age': [self.age],
                'systolic_bp': [self.systolic_bp],
                'cholesterol': [self.cholesterol],
                'gender': [self.gender],
                'diabetes': [self.diabetes],
                'smoking_status': [self.smoking_status],
                'chest_pain_type': [self.chest_pain_type],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe created')
            return df
        
        except Exception as e:
            logging.info('Error occured in get data as dataframe function in prediction pipeline')
            raise CustomException(e,sys)