import sys
import os
# Add the parent directory of "srouce" to sys.path if not already present
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataclasses import dataclass

from logger import logging
from exception import CustomException
from utils import save_function, model_performance

from sklearn.linear_model import LogisticRegression


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('datalog', 'model.pkl')


class ModelTrainer:
    def _init_(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Separating the dependent and independent variables")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1], test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),

            }

            model_report = model_performance(X_train, y_train, X_test, y_test, models)
            logging.info(f"Model Report: {model_report}")

            # Find best model based on F1 Score
            best_model_name = max(model_report, key=lambda k: model_report[k]["F1 Score"])
            best_model_score = model_report[best_model_name]["F1 Score"]
            best_model = models[best_model_name]

            print(f"The best model is {best_model_name}, with F1 Score: {best_model_score}")
            logging.info(f"The best model is {best_model_name}, with F1 Score: {best_model_score}")

            save_function(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)