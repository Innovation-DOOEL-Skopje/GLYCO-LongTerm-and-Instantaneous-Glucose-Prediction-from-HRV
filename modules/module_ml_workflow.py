import sys
from comet_ml import Experiment
from sklearn.metrics import classification_report
from datetime import datetime
from typing import Dict
import pandas as pd
import pickle
import json
import os
import module_results

#____________________________________________________________________________________________________________________________________


DATA_DIR_KEY = 'data'
MODEL_DIR_KEY = 'model'
CV_DIR_KEY = 'cv'
PLOTS_DIR_KEY = 'plots'
PICKLE_DIR_KEY = 'pickle'
JSON_DIR_KEY = 'json'
TEST_DIR_KEY = 'test'

#____________________________________________________________________________________________________________________________________


def clear_vars_packages() -> None:
    """Clear all loaded variables"""

    for name in dir():
        if not name.startswith('_'):
            del globals()[name]
#____________________________________________________________________________________________________________________________________

def get_workflow_dirs(which_pc: str = 'mine',
                      data_dir: str = None,
                      model_id: str = None,
                      num_iter: int = 0) -> Dict:

    """ Create a dictionary of directories which are needed for the ML workflow.
        Keys:   'data' - the data dir.
                'model' - the model dir.
                 sub: 'cv' - the cross validation results.
                 sub: 'plots' - the plots for the model.
                 sub: 'pickle' - the serializing directory.

    :param which_pc: the pc used
    :param data_dir: the data directory name
    :param model_id: model short discription i.e. ID
    :param num_iter: number of iterations of the hyparparam search

    :return: dict with the keys described and the appropriate dirs as their values
    """

    # get data and project root dir based on pc
    if which_pc.startswith('pano'):
        root_dir = f"C:\\Users\\Inno\\Documents\\IV\\GLYCO"
        data_dir = f"C:\\Users\\Inno\\Documents\\IV\\PREPROCESSED_DATA\\{data_dir}"
    else:
        root_dir = f"C:\\Users\\ilija\\Pycharm Projects\\GLYCO"
        data_dir = f"{root_dir}\\_DATA_PREPROCESSED\\{data_dir}"

    # define the model directory (if less than 10 iter it is only a debugging/test try)
    if num_iter < 10: model_id ='TRY_____' + model_id
    model_dir = f'{root_dir}\\RESULTS\\September\\{str(datetime.now().date())}\\' \
                f'{model_id}_{num_iter}_{str(datetime.now().hour)}h{str(datetime.now().minute)}m'

    # create the workflow directories dictionary with global constants
    #   that will be used throughout the module
    workflow_dirs_dict = {
        DATA_DIR_KEY: data_dir,
        MODEL_DIR_KEY: model_dir,
        CV_DIR_KEY: f'{model_dir}\\CV',
        PLOTS_DIR_KEY: f'{model_dir}\\Plots',
        PICKLE_DIR_KEY: f'{model_dir}\\Pickled_Models',
        JSON_DIR_KEY: f'{model_dir}\\JSON',
        TEST_DIR_KEY: f'{model_dir}\\Test',
    }

    # if debugging or prototyping, delete old dir with same name
    #   that is created not more than a minute ago
    if num_iter < 10 and os.path.exists(model_dir):
        os.remove(model_dir)

    # create the directories
    os.makedirs(model_dir)
    print(f'Created dir: {model_dir}')
    os.makedirs(workflow_dirs_dict[CV_DIR_KEY])
    print(f'Created dir: {workflow_dirs_dict[CV_DIR_KEY]}')
    os.makedirs(workflow_dirs_dict[PLOTS_DIR_KEY])
    print(f'Created dir: {workflow_dirs_dict[PLOTS_DIR_KEY]}')
    os.makedirs(workflow_dirs_dict[PICKLE_DIR_KEY])
    print(f'Created dir: {workflow_dirs_dict[PICKLE_DIR_KEY]}')
    os.makedirs(workflow_dirs_dict[JSON_DIR_KEY])
    print(f'Created dir: {workflow_dirs_dict[JSON_DIR_KEY]}')
    os.makedirs(workflow_dirs_dict[TEST_DIR_KEY])
    print(f'Created dir: {workflow_dirs_dict[TEST_DIR_KEY]}')

    return workflow_dirs_dict

#____________________________________________________________________________________________________________________________________


def train_and_save_sklearn(hyperparameters_optimizer,
                           train_test_sets:Dict = None,
                           submodel_id_dict: Dict = None,
                           workflow_dirs: Dict = None,
                           comet_dict: Dict = None,
                           ):

    X_train = train_test_sets['X_train']
    y_train = train_test_sets['y_train']

    # train model and save training time
    start_time = datetime.now()
    hyperparameters_optimizer.fit(X_train, y_train)
    end_time = datetime.now()
    training_time = end_time - start_time

    # save serialized model
    pickle.dump(hyperparameters_optimizer.best_estimator_,
                open(f'{workflow_dirs[PICKLE_DIR_KEY]}\\PICKLED_{submodel_id_dict["full_string"]}.sav', 'wb'))

    # save cross validation results
    pd.DataFrame(hyperparameters_optimizer.cv_results_)\
        .to_csv(f'{workflow_dirs[CV_DIR_KEY]}\\CV_{submodel_id_dict["full_string"]}.csv', index = False)

    for i in range(len(hyperparameters_optimizer.cv_results_['params'])):
        exp = Experiment(api_key = comet_dict['api'], project_name = comet_dict['project_name'],
                         auto_output_logging = "native")
        exp.add_tag(tag = comet_dict['tag'])
        for k, v in hyperparameters_optimizer.cv_results_.items():
            if k == "params":
                exp.log_parameters(v[i])
            else:
                exp.log_metric(k, v[i])

    # save model info in json format
    write_to_json_sklearn(submodel_id_dict, hyperparameters_optimizer, workflow_dirs)

    train_dict = dict(
        n_iter = hyperparameters_optimizer.n_iter,
        training_time = str(training_time),
        start_time = str(start_time.time())[:-7],
        end_time = str(end_time.time())[:-7]
    )

    return hyperparameters_optimizer.best_estimator_, train_dict

#____________________________________________________________________________________________________________________________________


def test_and_save_sklearn(best_model,
                          train_test_sets,
                          submodel_id_dict,
                          workflow_dirs
                          ):

    X_test = train_test_sets['X_test']
    y_test = train_test_sets['y_test']
    y_train = train_test_sets['y_train']
    id_test = train_test_sets['id_test']

    y_pred = best_model.predict(X_test)

    tested_dataframe = X_test.copy()
    tested_dataframe['y_test'] = y_test
    tested_dataframe['y_pred'] = y_pred
    tested_dataframe['Patient_ID'] = id_test

    pd.DataFrame(tested_dataframe).to_csv(f'{workflow_dirs[TEST_DIR_KEY]}\\{submodel_id_dict["full_string"]}.csv', index = False)

    scores_and_samples_dict = module_results.results_dict_from_confusion_matrix_with_ratios(y_test, y_pred, y_train)

    first_columns_dict = dict(
            Dataset = submodel_id_dict["file_name"],
            Domain = submodel_id_dict["domain"],
            Algorithm = submodel_id_dict["classifier"]
    )

    test_dict = dict(**first_columns_dict, **scores_and_samples_dict)

    return test_dict

#____________________________________________________________________________________________________________________________________


def write_to_json_sklearn(submodel_id_dict, hyperparameters_optimizer, workflow_dirs):
    # JSON detailed model information

    json_dict = dict(
            ID = submodel_id_dict["full_string"],
            Model = str(hyperparameters_optimizer.best_estimator_),
            Best_Parameters = str(hyperparameters_optimizer.best_params_),
            Best_Score = str(hyperparameters_optimizer.best_score_)
    )

    if submodel_id_dict["classifier"].startswith('lr'):
        json_dict['Coefficients'] = str(hyperparameters_optimizer.best_estimator_['logistic_regression'].coef_),
        json_dict['Intercept'] = str(hyperparameters_optimizer.best_estimator_['logistic_regression'].intercept_),

    if submodel_id_dict["classifier"].startswith('rf'):
        json_dict['Feature_Importances'] = str(hyperparameters_optimizer.best_estimator_['random_forest'].feature_importances_),

    json_file = json.dumps(json_dict, indent= 4)
    json_write_path = f'{workflow_dirs[JSON_DIR_KEY]}\\{submodel_id_dict["full_string"]}.json'
    f = open(json_write_path, "w")
    f.write(json_file)
    f.close()

#____________________________________________________________________________________________________________________________________

#
# def train_and_save_BayesSearchCV(hyperparameters_optimizer,
#                                  train_test_sets,
#                                  submodel_id_dict,
#                                  workflow_dirs,
#                                  ):
#
#     X_train = train_test_sets['X_train']
#     y_train = train_test_sets['y_train']
#     id_train = train_test_sets['id_train']
#
#     # train model and save training time
#     start_time = datetime.now()
#     hyperparameters_optimizer.fit(X_train, y_train, groups = id_train)
#     end_time = datetime.now()
#     training_time = end_time - start_time
#
#     # save serialized model
#     pickle.dump(hyperparameters_optimizer.best_estimator_,
#                 open(f'{workflow_dirs[PICKLE_DIR_KEY]}\\PICKLED_{submodel_id_dict["full_string"]}.sav', 'wb'))
#
#     # save cross validation results
#     pd.DataFrame(hyperparameters_optimizer.cv_results_)\
#         .to_csv(f'{workflow_dirs[CV_DIR_KEY]}\\CV_{submodel_id_dict["full_string"]}.csv', index = False)
#
#     # save model info in json format
#     # write_to_json_sklearn(submodel_id_dict, hyperparameters_optimizer, workflow_dirs)
#
#     train_dict = dict(
#         n_iter = hyperparameters_optimizer.n_iter,
#         training_time = str(training_time),
#         start_time = str(start_time.time())[:-7],
#         end_time = str(end_time.time())[:-7]
#     )
#
#     return hyperparameters_optimizer.best_estimator_, train_dict


#____________________________________________________________________________________________________________________________________


