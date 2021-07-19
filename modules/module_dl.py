import comet_ml

import tensorflow as tf

print(f'Using tensorflow version: {tf.version.VERSION}')

import keras
import keras.backend as K
from keras.layers import Dense, Dropout
from keras.metrics import TrueNegatives, TruePositives, FalseNegatives, FalsePositives

import pandas as pd
import numpy as np

from typing import List, Dict, Union

import warnings

from sklearn.metrics import confusion_matrix

VALID_CLASSIFICATION_LOSSES = ['BinaryCrossentropy', 'CategoricalCrossentropy', 'SparseCategoricalCrossentropy',
                               'Poisson', 'KLDivergence', 'Hinge', 'SquaredHinge', 'CategoricalHinge']

VALID_INITIALIZERS = ['Zeros', 'Ones', 'Identity', 'Orthogonal', 'Constant', 'VarianceScaling',
                      'TruncatedNormal', 'RandomNormal', 'GlorotNormal', 'RandomUniform', 'GlorotUniform']

VALID_OPTIMIZERS = ['Adam', 'Adadelta', 'Adamax', 'SGD', 'Ftrl', 'Nadam', 'RMSprop', 'Adagrad']

# ____________________________________________________________________________________________________________________________________


def check_name(input_name: str, compare_to_name: str) -> bool:
    if input_name.lower() == compare_to_name.lower():
        return True

# ____________________________________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________________________________


class DL:
    callbacks = list()
    fit_metadata = dict()

    def __init__(self,
                 split_sets: Dict[str, Union[np.ndarray]],  # , np.ndarray]],
                 split_sets_metadata: Dict,
                 model_type: str = 'sequential',
                 layers_list: List[Dict] = None,
                 hyper_params: Dict[str, Union[int, float, str]] = None,
                 hyper_method_names: Dict[str, str] = None,
                 early_stopping_flag: bool = True,
                 comet_experiment: comet_ml.BaseExperiment = None,
                 trained_models_dir= None,
                 model_id = None
                 ):
        # mutable defaults handling
        if layers_list is None:
            layers_list = [{'units': 32, 'type': 'dense', 'activation': 'relu'},
                           {'units': 64, 'type': 'dense', 'activation': 'relu'}]

        # CometML Experiment
        self.comet_experiment = comet_experiment

        # training and testing sets and their metadata
        self.split_sets = split_sets
        self.split_sets_metadata = split_sets_metadata

        # hyper parameters and methods
        self.hyper_params = hyper_params
        self.hyper_method_names = hyper_method_names
        self.hyper_methods = self.get_initialized_hyper_methods(hyper_method_names)

        # callbacks
        self.set_early_stopping(early_stopping_flag)

        # make model
        self.model = self.get_initialized_model(model_type)
        self.define_layers(layers_list);
        self.layers_list = layers_list  # keep just in case
        self.compile_model()

        # test_scores
        self.test_scores = dict()
        self.trained_models_dir = trained_models_dir
        self.model_id = model_id

    # ____________________________________________________________________________________________________________________________________

    def get_initialized_hyper_methods(self, hyper_method_names: Dict[str, str]) -> [str, keras]:
        hyper_methods = dict(
                loss = self.get_loss(hyper_method_names['loss']),
                initializer = self.get_initializer(hyper_method_names['initializer']),
                optimizer = self.get_optimizer(hyper_method_names['optimizer'])
        )

        return hyper_methods

    # ____________________________________________________________________________________________________________________________________

    @classmethod
    def get_initialized_model(cls, model_type) -> keras.models:
        if check_name(model_type, 'sequential'): return keras.models.Sequential()

        raise ValueError(f'Unknown model type! \nGot {model_type} as model_type.')

    # ____________________________________________________________________________________________________________________________________

    def define_layers(self, layers_list: List) -> None:
        first_layer_flag = True

        for layer_dict in layers_list:
            # first layer
            if first_layer_flag:
                first_layer_flag = False

                if check_name(layer_dict['type'], 'Dense'):
                    self.model.add(Dense(units = layer_dict['units'],
                                         input_shape = (self.split_sets['X_train'].shape[1],),
                                         activation = layer_dict['activation'],
                                         kernel_initializer = self.hyper_methods['initializer'],
                                         kernel_regularizer = self.hyper_method_names['regularizer']))

                if check_name(layer_dict['type'], 'Dropout'):
                    raise ValueError('First layer should not be a dropout layer!')

            # after first layer
            else:
                if check_name(layer_dict['type'], 'Dense'):
                    self.model.add(Dense(units = layer_dict['units'],
                                         activation = layer_dict['activation'],
                                         kernel_initializer = self.hyper_methods['initializer'],
                                         kernel_regularizer = self.hyper_method_names['regularizer']))

                if check_name(layer_dict['type'], 'Dropout'):
                    self.model.add(Dropout(rate = layer_dict['rate']))

    # ____________________________________________________________________________________________________________________________________

    def compile_model(self) -> None:
        self.model.compile(loss = self.hyper_methods['loss'],
                           optimizer = self.hyper_methods['optimizer'],
                           # metrics = ["accuracy", self.f1, self.precision, self.recall],
                           metrics = ["accuracy",
                                      # self.true_positives, self.true_negatives, self.false_positives, self.false_negatives,
                                      # self.f1_macro, self.f1_1, self.f1_0,
                                      # self.precision, self.recall, self.specificity, self.npv,
                                      # tf.keras.metrics.AUC(curve="ROC", name = 'ROC_AUC'), tf.keras.metrics.AUC(curve="PR", name = 'PR_AUC')
                                      ])

        print(self.model.summary())

    # ____________________________________________________________________________________________________________________________________

    def fit_model(self) -> None:
        if self.comet_experiment is not None:
            with self.comet_experiment.train():
                fit_metadata_tracker = self.model.fit(x = self.split_sets['X_train'],
                                                      y = self.split_sets['y_train'],
                                                      epochs = self.hyper_params['epochs'],
                                                      batch_size = self.hyper_params['batch_size'],
                                                      validation_split = self.hyper_params['validation_split'],
                                                      callbacks = self.callbacks,
                                                      verbose = 10,
                                                      class_weight = self.split_sets_metadata['class_weight'],
                                                      workers = -1
                                                      )

        else:
            fit_metadata_tracker = self.model.fit(x = self.split_sets['X_train'],
                                                  y = self.split_sets['y_train'],
                                                  epochs = self.hyper_params['epochs'],
                                                  batch_size = self.hyper_params['batch_size'],
                                                  validation_split = self.hyper_params['validation_split'],
                                                  callbacks = self.callbacks,
                                                  verbose = 10,
                                                  class_weight = self.split_sets_metadata['class_weight'],
                                                  workers = -1
                                                  )

        self.set_fit_metadata(fit_metadata_tracker)
        self.model.save(f'{self.trained_models_dir.path}\\{self.model_id}.h5')

    # ____________________________________________________________________________________________________________________________________

    def set_fit_metadata(self, fit_tracker) -> None:
        self.fit_metadata['n_epochs'] = len(fit_tracker.history['loss'])

    # ____________________________________________________________________________________________________________________________________

    def get_fit_num_epochs(self) -> int:
        return self.fit_metadata['n_epochs']

    # ____________________________________________________________________________________________________________________________________

    def test_model(self) -> Dict[str, float]:
        if self.comet_experiment is not None:
            with self.comet_experiment.test():
                # loss, tp, tn, fp, fn, accuracy, f1_macro, f1_1, f1_0, precision, recall, specificity, npv, roc_auc, pr_auc = \
                loss, accuracy = self.model.evaluate(self.split_sets['X_test'], self.split_sets['y_test'], verbose = 10,
                                                     batch_size = 1000000, workers = -1)

        else:
            # loss, tp, tn, fp, fn, accuracy, f1_macro, f1_1, f1_0, precision, recall, specificity, npv, roc_auc, pr_auc = \
            loss, accuracy = self.model.evaluate(self.split_sets['X_test'], self.split_sets['y_test'], verbose = 10,
                                                 batch_size = 1000000, workers = -1)

        self.test_scores['loss'] = loss
        self.test_scores['accuracy'] = accuracy
        # self.test_scores['f1_macro'] = f1_macro
        # self.test_scores['f1_1'] = f1_1
        # self.test_scores['f1_0'] = f1_0
        # self.test_scores['precision'] = precision
        # self.test_scores['recall'] = recall
        # self.test_scores['specificity'] = specificity
        # self.test_scores['npv'] = npv
        # self.test_scores['roc_auc'] = roc_auc
        # self.test_scores['pr_auc'] = pr_auc
        # self.test_scores['tp'] = tp
        # self.test_scores['tn'] = tn
        # self.test_scores['fp'] = fp
        # self.test_scores['fn'] = fn

        # if self.comet_experiment is not None:
        #
        #     # last logs and ending experiment
        #     self.comet_experiment.log_metrics(dict(logged_loss = loss,
        #                                            logged_accuracy = accuracy,
        #                                            logged_f1_macro = f1_macro,
        #                                            logged_f1_1 = f1_1,
        #                                            logged_f1_0 = f1_0,
        #                                            logged_precision = precision,
        #                                            logged_recall = recall,
        #                                            logged_specificity = specificity,
        #                                            logged_npv = npv,
        #                                            logged_roc_auc = roc_auc,
        #                                            logged_pr_auc = pr_auc,
        #                                            logged_tp = tp,
        #                                            logged_tn = tn,
        #                                            logged_fp = fp,
        #                                            logged_fn = fn,
        #                                            logged_balance_train = self.split_sets_metadata['class_weight'][1],
        #                                            logged_balance_test = self.split_sets_metadata['class_balance_test'][1],
        #                                            ))
        #
        #     self.comet_experiment.log_parameters({**self.hyper_params,
        #                                           **self.hyper_methods,
        #                                           'architecture': self.layers_list})
        #     self.comet_experiment.end()

        return self.test_scores

    # ____________________________________________________________________________________________________________________________________

    def predict(self, return_metrics: bool = False):
        if return_metrics:
            return self.metrics_report(y_pred = self.model.predict_classes(self.split_sets['X_test']),
                                       y_test = self.split_sets['y_test'])
        else:
            return self.model.predict_classes(self.split_sets['X_test'])

    # ____________________________________________________________________________________________________________________________________

    def get_optimizer(self, optimizer_name: str) -> keras.optimizers:
        if check_name(optimizer_name, 'Adam'): return keras.optimizers.Adam(
            learning_rate = self.hyper_params['learning_rate'],
            decay = self.hyper_params['decay'])
        if check_name(optimizer_name, 'Adadelta'): return keras.optimizers.Adadelta(
            learning_rate = self.hyper_params['learning_rate'])
        if check_name(optimizer_name, 'Adamax'): return keras.optimizers.Adamax(
            learning_rate = self.hyper_params['learning_rate'])
        if check_name(optimizer_name, 'SGD'): return keras.optimizers.SGD(
            learning_rate = self.hyper_params['learning_rate'])
        if check_name(optimizer_name, 'Ftrl'): return keras.optimizers.Ftrl(
            learning_rate = self.hyper_params['learning_rate'])
        if check_name(optimizer_name, 'Nadam'): return keras.optimizers.Nadam(
            learning_rate = self.hyper_params['learning_rate'])
        if check_name(optimizer_name, 'RMSprop'): return keras.optimizers.RMSprop(
            learning_rate = self.hyper_params['learning_rate'])
        if check_name(optimizer_name, 'Adagrad'): return keras.optimizers.Adagrad(
            learning_rate = self.hyper_params['learning_rate'])

        raise ValueError('Unknown optimizer!')

    # ____________________________________________________________________________________________________________________________________

    @classmethod
    def get_initializer(cls, initializer_name: str, initializer_kwargs: dict = None) -> keras.initializers:
        if initializer_kwargs is None:
            if check_name(initializer_name, 'Zeros'): return keras.initializers.Zeros()
            if check_name(initializer_name, 'Ones'): return keras.initializers.Ones()
            if check_name(initializer_name, 'Identity'): return keras.initializers.Identity()
            if check_name(initializer_name, 'Orthogonal'): return keras.initializers.Orthogonal()
            if check_name(initializer_name, 'Constant'): return keras.initializers.Constant()
            if check_name(initializer_name, 'TruncatedNormal'): return keras.initializers.TruncatedNormal()
            if check_name(initializer_name, 'RandomNormal'): return keras.initializers.RandomNormal()
            if check_name(initializer_name, 'GlorotNormal'): return keras.initializers.GlorotNormal()
            if check_name(initializer_name, 'RandomUniform'): return keras.initializers.RandomUniform()
            if check_name(initializer_name, 'GlorotUniform'): return keras.initializers.GlorotUniform()

            # if activtion is relu, as recommended by andrew yang; justification and research paper yet to be read
            if check_name(initializer_name, 'VarianceScaling'): return keras.initializers.VarianceScaling(scale = 2.0)

        raise ValueError('Unknown initializer!')

    # ____________________________________________________________________________________________________________________________________

    @classmethod
    def get_loss(cls, loss_name: str) -> keras.losses:
        # regression losses
        if check_name(loss_name, 'Huber'): return keras.losses.Huber()
        if check_name(loss_name, 'LogCosh'): return keras.losses.LogCosh()
        if check_name(loss_name, 'Reduction'): return keras.losses.Reduction()
        if check_name(loss_name, 'CosineSimilarity'): return keras.losses.CosineSimilarity()
        if check_name(loss_name, 'MeanSquaredError'): return keras.losses.MeanSquaredError()
        if check_name(loss_name, 'MeanAbsoluteError'): return keras.losses.MeanAbsoluteError()
        if check_name(loss_name, 'MeanSquaredLogarithmicError'): return keras.losses.MeanSquaredLogarithmicError()
        if check_name(loss_name, 'MeanAbsolutePercentageError'): return keras.losses.MeanAbsolutePercentageError()

        # probabilistic classification losses
        if check_name(loss_name, 'Poisson'): return keras.losses.Poisson()
        if check_name(loss_name, 'KLDivergence'): return keras.losses.KLDivergence()
        if check_name(loss_name, 'BinaryCrossentropy'): return keras.losses.BinaryCrossentropy()
        if check_name(loss_name, 'CategoricalCrossentropy'): return keras.losses.CategoricalCrossentropy()
        if check_name(loss_name, 'SparseCategoricalCrossentropy'): return keras.losses.SparseCategoricalCrossentropy()

        # max margin classification losses
        if check_name(loss_name, 'Hinge'): return keras.losses.Hinge()
        if check_name(loss_name, 'SquaredHinge'): return keras.losses.SquaredHinge()
        if check_name(loss_name, 'CategoricalHinge'): return keras.losses.CategoricalHinge()

        raise ValueError('Unknown loss!')

    # ____________________________________________________________________________________________________________________________________

    def set_early_stopping(self, early_stopping_flag: bool) -> None:
        if early_stopping_flag:
            self.callbacks.append(
                    keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                  mode = 'min',
                                                  verbose = 10,
                                                  patience = self.hyper_params['patience'])
            )

    # ____________________________________________________________________________________________________________________________________

    @classmethod
    def metrics_report(cls, y_test: np.ndarray, y_pred: np.ndarray):
        y_test = np.ravel(y_test)
        y_pred = np.ravel(y_pred)

        true_positives = np.sum(y_test * y_pred)
        false_positives = np.sum(np.abs(y_test - 1) * y_pred)
        true_negatives = np.sum((y_test - 1) * (y_pred - 1))
        false_negatives = np.sum(y_test * np.abs(y_pred - 1))

        accuracy = round(
            (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives),
            4)
        precision = round(true_positives / (true_positives + false_positives), 4)
        recall = round(true_positives / (true_positives + false_negatives), 4)
        specificity = round(true_negatives / (true_negatives + false_positives), 4)
        npv = round(true_negatives / (true_negatives + false_negatives), 4)
        f1_1 = round(2 * (precision * recall) / (precision + recall), 4)
        f1_0 = round(2 * (specificity * npv) / (specificity + npv), 4)
        f1_macro = round((f1_1 + f1_0) / 2, 4)

        return dict(
                Accuracy = accuracy, f1_macro = f1_macro,
                f1_1 = f1_1, f1_0 = f1_0,
                Precision = precision, Recall = recall,
                Specificity = specificity, npv = npv,
                TP = int(true_positives), FP = int(false_positives), FN = int(false_negatives),
                TN = int(true_negatives),
                y_test_shape = y_test.shape,
                y_pred_shape = y_pred.shape,
                total_samples = true_negatives + true_positives + false_positives + false_negatives,
        )

    # ____________________________________________________________________________________________________________________________________

    @classmethod
    def true_positives(cls, y_test, y_pred):
        return K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))

    @classmethod
    def false_positives(cls, y_test, y_pred):
        return K.sum(K.abs(y_test - 1) * y_pred)

    @classmethod
    def false_negatives(cls, y_test, y_pred):
        return K.sum(y_test * K.abs(y_pred - 1))

    @classmethod
    def true_negatives(cls, y_test, y_pred):
        return K.sum((y_test - 1) * (y_pred - 1))

    # ____________________________________________________________________________________________________________________________________

    @classmethod
    def precision(cls, y_test, y_pred):
        true_positives = K.sum(y_test * y_pred)
        false_positives = K.sum(K.abs(y_test - 1) * y_pred)
        return true_positives / (true_positives + false_positives + K.epsilon())

    @classmethod
    def recall(cls, y_test, y_pred):
        true_positives = K.sum(y_test * y_pred)
        false_negatives = K.sum(y_test * K.abs(y_pred - 1))
        return true_positives / (true_positives + false_negatives + K.epsilon())

    def f1_1(self, y_test, y_pred):
        precision = self.precision(y_test, y_pred)
        recall = self.recall(y_test, y_pred)
        return 2 * (precision * recall) / (precision + recall + K.epsilon())

    # ____________________________________________________________________________________________________________________________________

    @classmethod
    def specificity(cls, y_test, y_pred):
        true_negatives = K.sum((y_test - 1) * (y_pred - 1))
        false_positives = K.sum(K.abs(y_test - 1) * y_pred)

        return true_negatives / (true_negatives + false_positives + K.epsilon())

    @classmethod
    def npv(cls, y_test, y_pred):
        true_negatives = K.sum((y_test - 1) * (y_pred - 1))
        false_negatives = K.sum(y_test * K.abs(y_pred - 1))
        return true_negatives / (true_negatives + false_negatives + K.epsilon())

    def f1_0(self, y_test, y_pred):
        specificity = self.specificity(y_test, y_pred)
        npv = self.npv(y_test, y_pred)
        return 2 * (specificity * npv) / (specificity + npv + K.epsilon())

    # ____________________________________________________________________________________________________________________________________

    def f1_macro(self, y_test, y_pred):
        f1_1 = self.f1_1(y_test, y_pred)
        f1_0 = self.f1_0(y_test, y_pred)
        return (f1_1 + f1_0) / 2

    # ____________________________________________________________________________________________________________________________________

# ____________________________________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________________________________


class BinaryClassificationDL(DL):

    def __init__(self,

                 X_train: Union[np.ndarray, pd.DataFrame],
                 y_train: Union[np.ndarray, pd.DataFrame],
                 X_test: Union[np.ndarray, pd.DataFrame],
                 y_test: Union[np.ndarray, pd.DataFrame],

                 class_weight: Dict[int, float] = None,

                 epochs: int = 100,
                 batch_size: int = 100,
                 validation_split = 0.1,

                 learning_rate: float = 0.01,
                 decay = 1e-2,

                 loss_name: str = 'BinaryCrossentropy',
                 initializer_name: str = 'GlorotNormal',
                 optimizer_name = 'Adam',
                 regularizer: str = None,

                 early_stopping_flag: bool = True,
                 patience = 100,

                 layers_list: List[Dict] = None,
                 model_type: str = 'sequential',

                 comet_experiment: comet_ml.BaseExperiment = None,
                 trained_models_dir = None,
                 model_id: str = None

                 ):
        hyper_params = dict(
                decay = decay,
                epochs = epochs,
                patience = patience,
                batch_size = batch_size,
                learning_rate = learning_rate,
                validation_split = validation_split,
        )

        hyper_methods_names = dict(
                loss = loss_name,
                optimizer = optimizer_name,
                initializer = initializer_name,
                regularizer = regularizer,
        )

        split_sets = dict(
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test
        )
        split_sets = self.cast_to_numpy(split_sets)

        split_sets_metadata = dict()
        if class_weight is not None:
            split_sets_metadata['class_weight'] = class_weight
        else:
            split_sets_metadata['class_weight'] = self.set_class_weight(y_train)
        # split_sets_metadata['class_balance_test'] = self.set_class_weight(y_test)

        self.binary_classification_assertions(hyper_methods_names = hyper_methods_names)

        # parent constructor
        super(BinaryClassificationDL, self).__init__(split_sets = split_sets,
                                                     split_sets_metadata = split_sets_metadata,
                                                     model_type = model_type,
                                                     layers_list = self.validate_binary_classification_layers(
                                                         layers_list),
                                                     hyper_params = hyper_params,
                                                     hyper_method_names = hyper_methods_names,
                                                     early_stopping_flag = early_stopping_flag,
                                                     comet_experiment = comet_experiment,
                                                     trained_models_dir = trained_models_dir,
                                                     model_id=model_id)

    # ____________________________________________________________________________________________________________________________________

    @classmethod
    def cast_to_numpy(cls, split_sets):
        if type(split_sets['X_train']) is pd.DataFrame: split_sets['X_train'].to_numpy()
        if type(split_sets['X_test']) is pd.DataFrame: split_sets['X_test'].to_numpy()
        if type(split_sets['y_train']) is pd.DataFrame: split_sets['y_train'].to_numpy()
        if type(split_sets['y_test']) is pd.DataFrame: split_sets['y_test'].to_numpy()

        return split_sets

    # ____________________________________________________________________________________________________________________________________

    @classmethod
    def set_class_weight(cls, y: np.ndarray) -> Dict[int, float]:
        ones_count = np.count_nonzero(y)
        zero_count = y.shape[0] - np.count_nonzero(y)

        if ones_count > zero_count:
            class_weight = {0: 1.0,
                            1: ones_count / zero_count}
        else:
            class_weight = {0: zero_count / ones_count,
                            1: 1.0}

        return class_weight

    # ____________________________________________________________________________________________________________________________________

    @classmethod
    def binary_classification_assertions(cls, hyper_methods_names):
        assert hyper_methods_names[
                   'loss'] in VALID_CLASSIFICATION_LOSSES, "Passed loss is not intended for classification!"

    # ____________________________________________________________________________________________________________________________________

    @classmethod
    def validate_binary_classification_layers(cls, layers_list) -> List:
        if layers_list[-1]['units'] > 2:
            warnings.warn('Binary classifier ends with a layer with more than 2 units.'
                          'Appending a sigmoid classification layer as a last layer of the neural network!')
            layers_list.append({'units': 1, 'type': 'dense', 'activation': 'sigmoid'})

        if layers_list[-1]['units'] <= 2 and check_name(layers_list[-1]['activation'], 'relu'):
            warnings.warn(
                'Binary classifier ends with a layer no more than 2 units but does not have appropriate classification activation function'
                'Appending a sigmoid classification layer as a last layer of the neural network!')
            layers_list.append({'units': 1, 'type': 'dense', 'activation': 'sigmoid'})

        if layers_list[-1]['units'] == 2 and check_name(layers_list[-1]['activation'], 'sigmoid'):
            warnings.warn('Binary classifier ends with a layer with 2 units but a sigmoid activation function.'
                          'Changing the activation function to softmax!')

        if layers_list[-1]['units'] == 1 and check_name(layers_list[-1]['activation'], 'softmax'):
            warnings.warn('Binary classifier ends with a layer with 1 unit but a softmax activation function.'
                          'Changing the activation function t╤o sigmoid!')

        return layers_list