import sys
import os
from dataclasses import dataclass

# Add the parent directory of "srouce" to sys.path if not already present
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from exception import CustomException
from logger import logging
from utils import save_function

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('datalog', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')
            #Define which columns should be ordinal encoded and which should be scaled
            categorical_cols = ["gender","smoking_status", "diabetes","chest_pain_type", ]
            numerical_cols = ["age", "systolic_bp","cholesterol"]

            #Define the custom ranking for each ordinal variable
            gender_categories = ['Male', 'Female']
            smoking_status_categories =  ['Never', 'Former', 'Current']
            diabetes_categories =  ['No', 'Yes']
            chest_pain_type_categories=  ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']


            logging.info('Pipeline Initiated')

            #Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            #Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[gender_categories,smoking_status_categories,diabetes_categories,chest_pain_type_categories,])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            logging.info('Pipeline Completed')
            return preprocessor


        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            #Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')
            
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'heart_disease'

            input_feature_train_df= train_df.drop(columns='heart_disease',axis=1)
            target_feature_train= train_df[target_column_name].astype(int)
            

            input_feature_test_df = test_df.drop(columns='heart_disease',axis=1)
            target_feature_test_df = test_df[target_column_name].astype(int)

            #Transforming using preprocessor obj
            input_feature_train = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            #Applying SMOTE to Training data
            smote=SMOTE(random_state=42)
            input_feature_train_arr,target_feature_train_df=smote.fit_resample(input_feature_train,target_feature_train)

            logging.info('Applying preprocessing object on training and testing datasets')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_function(
                
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        

        except Exception as e:
            logging.info('Exception occured in the Intitiate Data Transformation stage')
            raise CustomException(e,sys)