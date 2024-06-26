# library doc string
'''
Purpose: This project is to show what has been learned through out the module for best coding practices.

Author: Anna Papadogiannakis
Date: May 18, 2024

'''
"""
import sklearn and panda libs
"""
# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    dataframe = pd.read_csv(pth)
    dataframe.head()
    return dataframe


def perform_eda(dataframe):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    #dataframe.shape
    dataframe.isnull().sum()
    dataframe.describe()

    dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    sns.set()
    plt.figure(figsize=(20, 10))
    dataframe["Churn"].hist()
    plt.savefig("./images/eda/churn_distribution.png")
    plt.figure(figsize=(20, 10))
    
    dataframe["Customer_Age"].hist()
    plt.savefig("./images/eda/customer_age_distribution.png")
    plt.figure(figsize=(20, 10))
    
    dataframe.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig("./images/eda/marital_status_distribution.png")
    plt.figure(figsize=(20, 10))
    
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(dataframe["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig("./images/eda/total_transaction_distribution.png")
    plt.figure(figsize=(20, 10))
    
    sns.heatmap(dataframe.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig("./images/eda/histogram.png")
    plt.close()


def encoder_helper(dataframe, category_lst, response=""):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument
            that could be used for naming variables or index y column]
    output:
            df: pandas dataframe with new columns for
    """
    for category in category_lst:
        category_col_lst = []
        category_groups = dataframe.groupby(category).mean()["Churn"]
        for val in dataframe[category]:
            category_col_lst.append(category_groups.loc[val])
        dataframe[f"{category}_Churn"] = category_col_lst
    return dataframe

def perform_feature_engineering(dataframe,response=""):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument
              that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"]
    x_data = pd.DataFrame()
    y_data = dataframe['Churn']
    dataframe = encoder_helper(dataframe, cat_columns,response)
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn"]

    x_data[keep_cols] = dataframe[keep_cols]
    #x_data = dataframe.drop("Churn", axis=1)
    #x_data.head()
    # This cell may take up to 15-20 minutes to run
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42
    )
    return (x_train, x_test, y_train, y_test)

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    # scores
    print("random forest results")
    print("test results")
    print(classification_report(y_test, y_test_preds_rf))
    print("train results")
    print(classification_report(y_train, y_train_preds_rf))
    print("logistic regression results")
    print("test results")
    print(classification_report(y_test, y_test_preds_lr))
    print("train results")
    print(classification_report(y_train, y_train_preds_lr))
    # Create plot
    plt.figure(figsize=(20,5))

    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train,
                                                  y_train_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.savefig("./images/results/rf_results.png")
    # approach improved by OP -> monospace!
    plt.axis('off')
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train,
                                                   y_train_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace')
    # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties = 'monospace')
    
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace')
    # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/logistics_results.png")
    plt.close()

def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    cv_rfc = model
    importances = cv_rfc.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    plt.figure()
    plt.bar(range(x_data.shape[1]), importances[names])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()

def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    # plots
    plt.figure(figsize=(15, 8))
    
    ax_gca = plt.gca()
    #rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=ax_gca, alpha=0.8)
    lrc_plot.plot(ax=ax_gca, alpha=0.8)
    plt.show()
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')
    lrc_plot = plot_roc_curve(lr_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax_gca = plt.gca()
    #rfc_disp = plot_roc_curve(rfc_model, x_test, y_test, ax=ax_gca, alpha=0.8)
    lrc_plot.plot(ax=ax_gca, alpha=0.8)
    plt.show()
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    plt.savefig("./images/results/roc_curve_result.png")
    feature_importance_plot(cv_rfc, x_test,
                            "./images/results/feature_importance.png")

if __name__ == "__main__":
    imported_dataframe = import_data(pth="./data/bank_data.csv")
    perform_eda(imported_dataframe)
    x_training, x_testing, y_training, y_testing = perform_feature_engineering(imported_dataframe)
    train_models(x_training, x_testing, y_training, y_testing)
