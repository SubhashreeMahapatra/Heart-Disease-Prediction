import sys
import os
# Add the parent directory of "srouce" to sys.path if not already present
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)