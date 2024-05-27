# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project is to show what has been learned through out the module for best coding practices.

## Files and data description
Overview of the files and data present in the root directory. 
```
data

|-bank_data.csv - input csv data

images

|-eda

|---churn_distribution.png - image generated from eda

|---customer_age_distribution.png - image generated from eda

|---heatmap.png - image generated from eda

|---marital_status_distribution.png - image generated from eda

|---total_transaction_distribution.png- image generated from eda

|-results

|---feature_importance.png- image generated feature importance

|---logistics_results.png- image generated from results

|---rf_results.png- image generated from results

|---roc_curve_result.png- image generated from results

logs

|-churn_library.log- log generated from running tests

models

|-logistic_model.pkl - logistics model generated 

|-rfc_model.pkl - rfc model generated 

churn_library.py - predict customer churn code

churn_notebook.ipynb - original notebook

churn_script_logging_and_tests.py - script tests

README.md

requirements_py3.6.txt - python 3.6 requirements

requirements_py3.8.txt - python 3.6 requirements
```

There is a images directory for outputed images by churn_library.py the  and logs directory outputed by churn_script_logging_and_tests.py

## Running Files

churn_library.py can be run directly via python and should run data processing and model creation for data in ./data/bank_data.csv.

Code Library dependency requirements are outlined in requirements_py3.6.txt

```
python -m pip install -r requirements_py3.6.txt
python churn_library.py
```

churn_script_logging_and_tests.py can be run via pytest to run tests for functions created in churn_library.py

```
pytest churn_script_logging_and_tests.py
```


