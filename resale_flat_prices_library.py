'''
overview

here are functions used to clean and engineer data of resale_flat_prices data from data.gov
clean columns such as converting flat_model to all uppercase for consistency
engineer new columns such as full_address and coordinates

there are also functions for modelling
such as cross validation of model and predicting using model

# download file from url and save as raw file
download_file(url, file_name)

# extract files in zip and read as dataframe
extract_csv_files_in_zip_to_dataframe(zip_file)

# combine, clean, and transform resale flat prices data
combine_clean_transform_zip(raw_data, coordinates_map, list_of_columns_to_use, start_date='1990-01-01', end_date='2100-01-01')

# combine, clean, and transform resale flat prices data
combine_clean_transform_zip_partial(raw_data, coordinates_map, list_of_columns_to_use=['year_month', 'block', 'street_name', 'full_address', 'latitude', 'longitude'])

# get coordinates from address as latitude and longitude using google geocode api
get_coordinates_from_address(address, api_key)

# clean coordinates
clean_coordinates(coordinates_df, coordinates_boundary)
    
# check if there are new addresses without coordinates in the coordinates_master file and update missing coordinates
check_and_update_address_coordinates(coordinates_map, clean_resale_flat_prices, api_key, coordinates_boundary)
    
# validate model using kfolds cross validation
cv_results(model, X, y, num_folds, kfold_random_state, to_print=True)
   
# predict using model and print mean absolute error and root mean squared error
pred_model(model, X_test, y_test, to_print=True)
   
# filter and split data into train and test set
filter_and_split_data(clean_data, test_months, train_years)
   
# train, validate, and test model
train_validate_test_model(train_data, test_data, train_years, model, random_state=42)
    
# define optuna objective for lgm model using cv
optuna_objective_lightgbm_cv(trial, train_data)

# optimise study with cross validation
optimise_study_cv(n_trials, train_data)

# train xgb model
train_xgbmodel(train_data, xgb_params=None)

# get feature importance of xgb model
get_feature_importance_from_xgb(xgb_model, X)

# plot feature importance from xgb model
plot_feature_importance_from_xgb(xgb_feature_importance, figsize, file_name='False')

'''



# imports

import requests
import shutil
import json
import zipfile

import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import requests

import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_validate



# download file from url and save as raw file
def download_file(url, file_name):
    '''
    download file from link and save to the same folder

    arguments:
    url (str): download url
    file_name (str): file name to save as, e.g. 'download.zip'
    '''
    # steam request
    with requests.get(url, stream=True) as r:
        # open file
        with open(file_name, 'wb') as f:
            # copy to same folder
            shutil.copyfileobj(r.raw, f)



# extract files in zip and read as dataframe
def extract_csv_files_in_zip_to_dataframe(zip_file):
    '''
    unzip file and loop through and read all csv files and combine into one dataframe

    arguments:
    zip_file (zip file): downloaded zip file

    retuns:
    a dataframe with all csv files
    '''
    # read zip file
    zf = zipfile.ZipFile(zip_file)
    # create empty dataframe to store data
    data_raw = pd.DataFrame()
    # loop through zip file and read each file as csv and concat into one dataframe
    for file in zf.namelist():
        # check if files' extension is .csv
        if file.endswith('.csv'):
            # read file as dataframe
            temp_data = pd.read_csv(zf.open(file))
            # concat temp data to existing data
            data_raw = pd.concat([data_raw, temp_data])
    
    # return
    return data_raw



# combine, clean, and transform resale flat prices data
def combine_clean_transform_zip(raw_data, coordinates_map, list_of_columns_to_use, start_date='1990-01-01', end_date='2100-01-01'):
    '''
    combine, clean, and transform resale flat prices data
    combine multiple dataframes together
    filter date range of data to be used
    clean 
        flat_model
        flat_type
        street_name
    create 
        floor from storey_range,
        remaining_lease_years from lease_commence_date
        coordinates
    select columns to be used for analysis

    arguments:
    list_of_flat_prices_df (list): list of dataframes of resale flat prices
    coordinates_map (df): dataframe of address and coordinates to map coordinates to address
    list_of_columns_to_use (list): list of columns to used for analysis
    start_date (str): default to '1990-01-01', start date of date range to be used for analysis in yyyy-mm-dd format, e.g. '1990-01-01' 
    end_date (str): default to '2100-01-01', end date of date range to be used for analysis in yyyy-mm-dd format, e.g. '2021-06-01'

    returns:
    a dataframe with columns from list_of_columns_to_use
    '''
    # copy data
    flat_prices_clean = raw_data.copy()

    # convert month to datetime format
    flat_prices_clean['year_month'] = pd.to_datetime(flat_prices_clean['month'])
    # filter max df date for analysis
    flat_prices_clean = flat_prices_clean.loc[(flat_prices_clean['year_month'] >= start_date) & (flat_prices_clean['year_month'] <= end_date)]

    # clean and engineer flat_prices_clean data
    # convert flat_model to upper for consistency
    flat_prices_clean['flat_model'] = flat_prices_clean['flat_model'].str.upper()
    # convert 'MULTI-GENERATION' and 'MULTI GENERATION' to 'MULTI GENERATION' for consistency
    flat_prices_clean['flat_type'] = flat_prices_clean['flat_type'].str.replace('-', ' ')
    # replace 'C'WEALTH' with 'COMMONWEALTH' in street_name for better results when fetching coordinates
    flat_prices_clean['street_name'] = flat_prices_clean['street_name'].str.replace('C\'WEALTH', 'COMMONWEALTH')

    # get max floor of storey_range as new 'floor' column and covert to int64
    # 'floor' will become a numerical data, a higher floor is typically considered to be better.
    flat_prices_clean['floor'] = flat_prices_clean['storey_range'].str[-2:].astype('int64')
    # create new column remaining_lease_years to show as number of years only
    # calculated as 99 years - (current year minus lease_commence_date) as hdbs are 99 year lease
    flat_prices_clean['remaining_lease_years'] = 99 - (dt.datetime.now().year - flat_prices_clean['lease_commence_date'])
    # merge coordinates on address
    # create full address from block and street_name
    flat_prices_clean['full_address'] = flat_prices_clean['block'] + ' ' + flat_prices_clean['street_name'] + ' SINGAPORE'

    # merge latitude and longitude
    flat_prices_clean = pd.merge(flat_prices_clean, coordinates_map[['full_address', 'latitude', 'longitude']], how='left', on='full_address')
    # drop records without coordinates
    flat_prices_clean = flat_prices_clean.loc[(flat_prices_clean['latitude'].notnull())]
    # select columns to use
    flat_prices_clean = flat_prices_clean[list_of_columns_to_use]

    # return
    return flat_prices_clean



# combine, clean, and transform resale flat prices data
def combine_clean_transform_zip_partial(raw_data, coordinates_map, list_of_columns_to_use=['year_month', 'block', 'street_name', 'full_address', 'latitude', 'longitude']):
    '''
    partial verison of combine_clean_transform_zip() to run on pythonanywhere to refresh data
    
    combine, clean, and transform resale flat prices data
    combine multiple dataframes together
    clean 
        street_name
    create 
        coordinates
    select columns to be used for analysis

    arguments:
    raw_data (list): dataframes of resale flat prices
    coordinates_map (df): dataframe of address and coordinates to map coordinates to address
    list_of_columns_to_use (list): list of columns to used cleaning

    returns:
    a dataframe with columns from list_of_columns_to_use
    '''
    # copy data
    flat_prices_clean = raw_data.copy()

    # convert month to datetime format
    flat_prices_clean['year_month'] = pd.to_datetime(flat_prices_clean['month'])
    # clean and engineer flat_prices_clean data
    # replace 'C'WEALTH' with 'COMMONWEALTH' in street_name for better results when fetching coordinates
    flat_prices_clean['street_name'] = flat_prices_clean['street_name'].str.replace('C\'WEALTH', 'COMMONWEALTH')
    # merge coordinates on address
    # create full address from block and street_name
    flat_prices_clean['full_address'] = flat_prices_clean['block'] + ' ' + flat_prices_clean['street_name'] + ' SINGAPORE'

    # get unique address
    flat_prices_clean = flat_prices_clean.drop_duplicates('full_address')

    # merge latitude and longitude
    flat_prices_clean = pd.merge(flat_prices_clean, coordinates_map[['full_address', 'latitude', 'longitude']], how='left', on='full_address')
    # select columns to use
    flat_prices_clean = flat_prices_clean[list_of_columns_to_use]

    # return
    return flat_prices_clean



 # get coordinates from address as latitude and longitude using google geocode api
def get_coordinates_from_address(address, api_key):
    '''
    get coodinates from an address using google geocode api
    information on how to set up and create api key can be found here
    https://developers.google.com/maps/documentation/geocoding/overview?hl=en_GB

    arguments:
    address (str): address to get coordinates of
    api_key (str): api key from google cloud platform

    returns:
    a tuple containing latitude and longitude
    '''
    ### api call ###
    # request response from google geocode api
    api_response = requests.get(f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}').json()
    # check if api response is 'OK'
    if api_response['status'] == 'OK':
        # get latitude from response
        latitude = api_response['results'][0]['geometry']['location']['lat']
        # get longitude from response
        longitude = api_response['results'][0]['geometry']['location']['lng']
    else:
        # if status is not 'OK', add status as error message
        latitude = 'error: ' + api_response['status']
        longitude = 'error: ' + api_response['status']

    # return a tuple
    return (latitude, longitude)



# clean coordinates
def clean_coordinates(coordinates_df, coordinates_boundary):
    '''
    clean coordinates by removing results from geocode api with errors
    remove coordinates that do not fall within a defined boundary, e.g. coordinates should not be outside of singapore

    arguments:
    coordinates_df (df): dataframe of address and coordinates after applying get_coordinates_from_address()
    coordinates_boundary (dict): nested dictionary of boundary of selected country with minimum and maximum latitude and longitude e.g. coordinates_boundary['SG']

    returns:
    a dataframe
    '''
    # copy data
    coordinates_clean = coordinates_df.copy()

    # split coodinates to latitude and longitude
    coordinates_clean['latitude'], coordinates_clean['longitude'] = zip(*coordinates_clean['coordinates'])
    # remove records where there are errors in coordinates
    coordinates_clean = coordinates_clean.loc[~(coordinates_clean['latitude'].astype(str).str.contains('error'))]
    # convert latitude and longitude to numeric
    coordinates_clean[['latitude', 'longitude']] = coordinates_clean[['latitude', 'longitude']].apply(pd.to_numeric)
    # filter coordinates
    coordinates_clean = coordinates_clean.loc[
        (coordinates_clean['latitude'] >= coordinates_boundary['min_lat']) &
        (coordinates_clean['latitude'] <= coordinates_boundary['max_lat']) &
        (coordinates_clean['longitude'] >= coordinates_boundary['min_lon']) &
        (coordinates_clean['longitude'] <= coordinates_boundary['max_lon'])
        ]

    # return
    return coordinates_clean



# check if there are new addresses without coordinates in the coordinates_master file and update missing coordinates
def check_and_update_address_coordinates(coordinates_map, clean_resale_flat_prices, api_key, coordinates_boundary):
    '''
    check clean_resale_flat_prices for addresses where 'latitude' is null
    get coordinates for addresses
    filter and keep only coordinates within the specified boundary
    add and update new coordinates to current coordinates_map and save as master file

    arguments:
    coordinates_map (df): current master file for coordinates and addresses
    clean_resale_flat_prices(df): clean resale flat prices data
    api_key (str): google maps geocode api key
    coordinates_boundary (dict): dictionary of boundary of singapore, using maximum and minimum latitude and longitude

    returns:
    a dataframe of an updated coordinates map
    '''
    # filter records where latitude/longitude is null
    missing_coords = clean_resale_flat_prices.loc[(clean_resale_flat_prices['latitude'].isnull() )]
    # select columns needed for getting coordinates
    missing_coords = missing_coords[['block', 'street_name', 'full_address']]
    # get unique address
    missing_coords = missing_coords.drop_duplicates()

    ### api call ###
    # apply get_coordinates_from_address() on full_addresss and store results in new column coordinates
    missing_coords['coordinates'] = missing_coords['full_address'].apply(lambda x: get_coordinates_from_address(x, api_key))

    # clean coordinates
    missing_coords = clean_coordinates(missing_coords, coordinates_boundary)
    # concat missing_coords with existing mapping file
    updated_coordinates_map = pd.concat([coordinates_map, missing_coords])

    # return
    return updated_coordinates_map



# validate model using kfolds cross validation
def cv_results(model, X, y, num_folds, kfold_random_state, to_print=True):
    '''
    validate model using kfolds cross validation

    arguments:
    model (model): model to be used for prediction
    X (df): dataframe of independent variables
    y (df): dataframe of dependent variables
    num_folds (int): number of folds to use
    to_print (bool): default to True, if True, print metrics

    returns:
    printed statement
    '''
    # define number of folds and shuffle data
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=kfold_random_state)
    # apply cross validation to return train/test score of mae and rmse
    cvresults = cross_validate(model, X, y, cv=kf, return_train_score=True, scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error'))
    # print metrics
    if to_print == True:
        # print model hyperparameters
        print(f'model name: {model}')
        # print mean train mae
        print('Train MAE: {}'.format(abs(cvresults['train_neg_mean_absolute_error'].mean())))
        # print mean validation mae
        print('Validation MAE: {}'.format(abs(cvresults['test_neg_mean_absolute_error'].mean())))
        # print mean train rmse
        print('Train RMSE: {}'.format(abs(cvresults['train_neg_root_mean_squared_error'].mean())))
        # print mean validation rsme
        print('Validation RMSE: {}'.format(abs(cvresults['test_neg_root_mean_squared_error'].mean())))

    # return
    return cvresults



# predict using model and print mean absolute error and root mean squared error
def pred_model(model, X_test, y_test, to_print=True):
    '''
    predict using model and print metrics

    arguments:
    model (model): fitted model to be used for prediction
    X_test (df): dataframe of independent variables from test data
    y_test (series): series of the dependent variable from test data
    to_print (bool): default to True, if True, print metrics

    returns:
    printed statement and predictions, mae, rmse
    '''
    # predict using trained model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    # print metrics
    if to_print == True:
        # calculate and print mae
        print('Test MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
        # caclulate and print rmse
        print('Test RMSE: {}'.format(mean_squared_error(y_test, y_pred, squared=False)))

    # return
    return y_pred, mae, rmse



# filter and split data into train and test set
def filter_and_split_data(clean_data, test_months, train_years):
    '''
    filter clean data into train and test set and train model
    define number of latest months of data to use as test data. 
    e.g. if max data date is August 2021 and defined number of months is 3, June, July August 2021 data will be set aside as test data
    define number of years of data to use as train data
    e.g. after the test data has been set aside and defined number of years is 2, May 2019 to May 2021 data will used as training data
    select variables to be used for modelling

    arguments:
    clean_data (df): cleaned data from combine_clean_transform_zip()
    test_months (int): number of months of data to set aside as test data
    train_years (int): number of years of data to set aside as training data

    returns:
    two dataframes, one train data set and one test dataset
    '''
    # create max train date as datetime
    max_train_date = clean_data['year_month'].max() - relativedelta(months=test_months)
    # create min train date as datetime
    min_train_date = max_train_date - relativedelta(years=train_years)

    # filter test data based on test_months
    test_data = clean_data.loc[(clean_data['year_month'] > max_train_date)]
    # filter train data based on train_years
    train_data = clean_data.loc[(clean_data['year_month'] >= min_train_date) & (clean_data['year_month'] <= max_train_date)]
    # select data for modelling
    columns_for_model = ['latitude', 'longitude', 'floor_area_sqm', 'floor', 'remaining_lease_years', 'resale_price']
    train_data = train_data[columns_for_model]
    test_data = test_data[columns_for_model]

    # return
    return train_data, test_data



# train, validate, and test model
def train_validate_test_model(train_data, test_data, train_years, model, random_state=42):
    '''
    machine learning model is trained and validated using k-folds cross validation with mean absolute error as the evaluation metric
    stores results of train, validation, train_validation_difference, test as dataframe
    storing the results as a dataframe provides the ability to run and trial multiple train_years and concat the results for comparison

    arguments:
    train_data (df): train data from filter_and_split_data()
    test_data (df): test data from filter_and_split_data()
    train_years (int): number of years of data to set aside as training data
    model (model): machine learning model
    random_state (int): default to 42, set a random state to return consistent results


    returns:
    trained xgb model and cross validation results
    or if mae_results is True then return
    a dataframe with train, validation, train_validation_difference, test mean absolute error
    '''
    # set X, y
    X = train_data.drop('resale_price', axis=1)
    y = train_data['resale_price']

    # fit training data
    model.fit(X, y)
    # set kfolds with 5 splits and shuffle data
    model_cv_results = cv_results(model, X, y, 5, random_state, to_print=False)
    # use trained model to predict unseen test data
    model_predictions, model_mae, model_rmse = pred_model(model, test_data.drop('resale_price', axis=1), test_data['resale_price'], to_print=False)

    # store mae as dataframe
    # includes train, validation, train_validation_difference, test results
    results_mae = pd.DataFrame({
        'train_years':[train_years], 
        'train_mae':[abs(model_cv_results['train_neg_mean_absolute_error'].mean())],
        'validation_mae':[abs(model_cv_results['test_neg_mean_absolute_error'].mean())],
        'train_validation_mae_difference': [abs(model_cv_results['test_neg_mean_absolute_error'].mean()) - abs(model_cv_results['train_neg_mean_absolute_error'].mean())],
        'test_mae':[model_mae]
        })

    # return
    return model, results_mae



# define optuna objective for lgm model using cv
def optuna_objective_lightgbm_cv(trial, train_data):
    '''
    define train data and convert to Dataset for lgb
    define optuna objective to tune hyperparameters of lightgbm models
    define hyperparameters to trial
    add pruning to prune trials that under perform i.e. mae of trial is worse than previous trials, speeding up the optimisation process
    cross validate lgb model on train data

    arguments:
    train_data (df): training data set

    returns:
    cv results of mean absolute error of lgb model on train data
    '''
    # set X, y
    X = train_data.drop('resale_price', axis=1)
    y = train_data['resale_price']
    # convert data to Dataset object for lgb
    d_train = lgb.Dataset(X, label=y)

    # set params
    param = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'random_state': 42,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 2, 200), # small num_leaves to avoid overfitting
        'max_depth': trial.suggest_int('max_depth', 1, 20), # small max_depth to avoid overfitting
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 1.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 1.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-5, 1.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100)
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'l1')
    # cross validate model
    cv_results = lgb.cv(param, d_train, stratified=False, verbose_eval=False, callbacks=[pruning_callback])
    # get mean absolute error
    cv_results = cv_results['l1-mean'][-1]

    # return
    return cv_results



# optimise study with cross validation
def optimise_study_cv(n_trials, train_data):
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize")
    objective = lambda trial: optuna_objective_lightgbm_cv(trial, train_data)
    study.optimize(objective, n_trials=n_trials)
    # return
    return study



# train xgb model
def train_xgbmodel(train_data, xgb_params=None):
    '''
    partial verison of train_validate_test_xgbmodel() to run on pythonanywhere to refresh model
    set X and y from training data and fit to xgb model with optional defined hyperparameters

    arguments:
    train_data (df): train data from filter_and_split_data()
    xgb_params (dict): default to None, pass a dictionary of hyperparameters to xgb model

    returns:
    trained xgb model
    '''
    # set X, y
    X = train_data.drop('resale_price', axis=1)
    y = train_data['resale_price']

    # create xgb model and pass any hyperparameter options
    xgb_model = xgb.XGBRegressor(**xgb_params)
    # fit training data
    xgb_model.fit(X, y)

    # return
    return xgb_model



# get feature importance of xgb model
def get_feature_importance_from_xgb(xgb_model, X):
    '''
    get feature importance of weight and gain of features used in xgb model as one dataframe

    arguments:
    xgb_model (model): fitted xgb model
    X (df): dataframe of dependent variables to extract features

    returns:
    a dataframe with each feature's weight and gain
    '''
    # create dataframe with features as column
    xgb_feature_importance = pd.DataFrame({'features':list(X)})

    # get feature importances
    xgb_feature_importance_weights = xgb_model.get_booster().get_score(importance_type='weight')
    xgb_baseline_importance_gain = xgb_model.get_booster().get_score(importance_type='gain')

    # map feature importances to dataframe
    xgb_feature_importance['weight'] = xgb_feature_importance['features'].map(xgb_feature_importance_weights)
    xgb_feature_importance['gain'] = xgb_feature_importance['features'].map(xgb_baseline_importance_gain)

    # return 
    return xgb_feature_importance



# plot feature importance from xgb model
def plot_feature_importance_from_xgb(xgb_feature_importance, figsize, file_name='False'):
    '''
    plot feature importance of weight and gain from xgb model

    arguments:
    xgb_feature_importance (df): a dataframe of features and their weight and gain from get_feature_importance_from_xgb()
    figsize (tuple): a tuple of figsize to adjust, e.g. (20,5)
    file_name (str): default as 'False', if file_name entered then save file as file_name

    returns:
    show plot inline with optional saved file
    '''
    # set subplot sizes
    plt.subplots(figsize=(figsize))

    # weight of features
    plt.subplot(1, 2, 1)
    # plot barplot
    sns.barplot(data=xgb_feature_importance, x='weight', y='features')
    # set title
    plt.title('XGB Feature Importance: Weight')

    # gain of features
    plt.subplot(1, 2, 2)
    # plot barplot
    sns.barplot(data=xgb_feature_importance, x='gain', y='features')
    # set title
    plt.title('XGB Feature Importance: Gain')

    # save file if file_name has been entered
    if file_name != 'False':
        plt.savefig(file_name + '.png')

    # show plot
    plt.show()