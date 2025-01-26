from models.ModelTrainer import ModelTrainer
from models.ModelSaverLoader import ModelSaverLoader
from data.DataExtractor import DataExtractor
from data.DataManager import DataManager
from typing import List
from models.ModelFactory import ModelFactory
from decorators.ArgsChecker import ArgsChecker
import pandas as pd
from datetime import datetime


class ModelTrainStage:
    @ArgsChecker((None, DataManager, ModelSaverLoader, List[str]), None)
    def __init__(
        self,
        training_and_test_data_manager: DataManager,
        saver_loader: ModelSaverLoader,
        model_types: List[str],
    ):
        self.t_a_t_m = training_and_test_data_manager
        self.saver_loader = saver_loader
        self.model_types = model_types
        self.models_initialized = False
        self.models = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def run(self, symbol, subdir):
        if not self.models_initialized or self.subdir != subdir:
            self.models_initialized = False
            self.subdir = subdir

        if not self.models_initialized:
            d = self.saver_loader.check_existing_models(self.model_types, subdir)
            if d:
                self.models = self.saver_loader.load_models(self.model_types, subdir)
                print("Loaded existing models for retraining")
            else:
                print("新規でモデルを作成します")
                self.models = ModelFactory.create_models(self.model_types)
                print("Created new models for training")

            self.models_initialized = True

        full_data = self.t_a_t_m.load_data(symbol)
        if full_data.empty:
            print(f" {symbol} をスキップします")
            return

        correct_count = full_data[full_data["label"] == 1].shape[0]
        incorrect_count = full_data[full_data["label"] == 0].shape[0]
        ratio_tt = round(incorrect_count / correct_count, 1)
        print(f"training and testing... [correct : incorrect = 1 : {ratio_tt}]")
        print(f"correct:{correct_count}, incorrect:{incorrect_count}")

        self.X_train, self.X_test, self.y_train, self.y_test = (
            DataExtractor.extract_data(full_data)
        )

        start_date = pd.to_datetime(full_data["date"]).min().strftime("%Y-%m-%d")
        end_date = pd.to_datetime(full_data["date"]).max().strftime("%Y-%m-%d")
        years_difference = (
            pd.to_datetime(end_date) - pd.to_datetime(start_date)
        ) / pd.Timedelta(days=365)
        print(
            f"Training period: {start_date} to {end_date} （約{years_difference:.1f}年間修行）"
        )

        self.models, results_df = ModelTrainer.train(
            self.models, self.X_train, self.y_train, self.X_test, self.y_test
        )

        print(results_df)
        self.saver_loader.save_models(self.models, subdir)

        # print("Model Pipeline completed successfully")
