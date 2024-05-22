'''
Import required modules for testing
'''
import os
import logging
import pandas as pd
import churn_library as cls
import pytest

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return dataframe


def test_eda(dataframe):
    '''
    test perform eda function
    '''
    try:
        cls.perform_eda(dataframe)
        assert os.path.isfile("./images/histogram.png")
        logging.info("Testing perform_eda: SUCCESS - histogram plot created.")
    except AssertionError as err:
        logging.error("Testing perform_Eda: histogram plot not created")
        raise err


def test_perform_feature_engineering(dataframe):
    '''
    test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            dataframe)
        assert len(x_train)==(round(len(dataframe) * (1-0.3)))
        assert len(x_test)==(round(len(dataframe) * 0.3))
        assert len(y_train)==(round(len(dataframe) * (1-0.3)))
        assert len(y_test)==round(len(dataframe) * 0.3)

        logging.info("Testing encoder_helper: SUCCESS - has the correct number of rows")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Test has the wrong number of rows ")
        raise err


def test_encoder_helper(dataframe):
    '''
    test encoder_helper
    '''
    try:
        cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"]
        dataframe = cls.encoder_helper(dataframe, cat_columns)
        #print(dataframe.columns)
        for category in cat_columns:
            assert f"{category}__Churn" in list(dataframe.columns)
        logging.info("Testing encoder helper: SUCCESS - category columns exists")
    except AssertionError as err:
        logging.error("Testing encoder helper: category columns are missing")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(train_models)

        cls.train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile("./images/classification_results.png")
        logging.info("Testing train_models: SUCCESS - classification_results.png is created")

        assert os.path.isfile("./models/rfc_model.pkl")
        logging.info("Testing train_models: SUCCESS - rfc_model.pkl is created")
    except AssertionError as err:
        logging.error("Testing train_models: model and classification results are missing")
        raise err

@pytest.fixture(scope="module")
def dataframe():
    return cls.import_data("./data/bank_data.csv")
        
if __name__ == "__main__":
    imported_dataframe = test_import()
    test_eda(imported_dataframe)
    test_encoder_helper(imported_dataframe)
    test_perform_feature_engineering(imported_dataframe)
    #train_models(x_training, x_testing, y_training, y_testing)
    