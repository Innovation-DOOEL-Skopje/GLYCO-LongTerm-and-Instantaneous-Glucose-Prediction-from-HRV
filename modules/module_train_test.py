from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from typing import Union

#____________________________________________________________________________________________________________________________________


def find_best_group_split(dataframe: pd.DataFrame,
                          target_feature: str,
                          group_by_feature: str,
                          balance_focus: str = "train",
                          train_balance_threshold: float = None,
                          test_balance_treshold: float = None,
                          test_size: float = 0.2,
                          num_splits_to_try: int = 100,
                          random_state: Union[str, int] = 42
                          ) -> (pd.DataFrame, pd.DataFrame):

    """ Find the best split into train and test data, based on grouping and balance focus.

    :param random_state:
    :param dataframe: dataframe to split for max balance
    :param target_feature: name of the target feature of the dataset
    :param group_by_feature: name of the feature on which to group sets, preventing data leakage when needed
    :param balance_focus:  {'train', 'test'}
        Whether to focus on balancing the train or test set
    :param test_size: size of test set
    :param num_splits_to_try: how many splits to iterate through

    :return: pandas.Dataframe, pandas.Dataframe, float, float
         returns train, test,
    """
    valid_balance_focus_options = ["train", "test", "both", 'test_with_train_threshold', 'train_with_test_threshold']
    if balance_focus not in valid_balance_focus_options:
        raise ValueError(f"balance_focus parameter must be one of {valid_balance_focus_options}")

    if type(random_state) is str:
        import random
        random_state = random.randint(0, 1000)

    min_train_diff = 1
    min_test_diff = 1
    min_train_indices = list()
    min_test_indices = list()
    # using GroupShuffleSplit to generate 10 splits (the golden rule) and find the best split for our goal
    for train_indices, test_indices in GroupShuffleSplit(test_size = test_size,
                                                         n_splits = num_splits_to_try,
                                                         random_state = random_state
                                                         ).split(dataframe.drop(target_feature, axis = 1),
                                                                 dataframe[target_feature],
                                                                 groups=dataframe[group_by_feature]):

        train, test = dataframe.iloc[train_indices].copy(), dataframe.iloc[test_indices].copy()

        vc_train = dict(train[target_feature].value_counts())
        if vc_train.get(0) is None or vc_train.get(1) is None: continue
        n_train = vc_train.get(0) + vc_train.get(1)
        zero_train, target_train = vc_train.get(0) / n_train, vc_train.get(1) / n_train
        train_balance = max(vc_train.get(0), vc_train.get(1)) / min(vc_train.get(0), vc_train.get(1))

        vc_test = dict(test[target_feature].value_counts())
        if vc_test.get(0) is None or vc_test.get(1) is None: continue
        n_test = vc_test.get(0) + vc_test.get(1)
        zero_test, target_test = vc_test.get(0) / n_test, vc_test.get(1) / n_test
        test_balance = max(vc_test.get(0), vc_test.get(1)) / min(vc_test.get(0), vc_test.get(1))

        if len(min_train_indices) == 0 and len(min_test_indices) == 0:
            min_train_diff = abs(zero_train - target_train)
            min_test_diff = abs(zero_test - target_test)
            min_train_indices = train_indices
            min_test_indices = test_indices
        elif balance_focus == 'train' and abs(zero_train - target_train) < min_train_diff:
            min_train_diff = abs(zero_train - target_train)
            min_test_diff = abs(zero_test - target_test)
            min_train_indices = train_indices
            min_test_indices = test_indices
        elif balance_focus == 'test' and abs(zero_test - target_test) < min_test_diff:
            min_train_diff = abs(zero_train - target_train)
            min_test_diff = abs(zero_test - target_test)
            min_train_indices = train_indices
            min_test_indices = test_indices
        elif balance_focus == 'both' and abs(zero_test - target_test) < min_test_diff and abs(zero_train - target_train) < min_train_diff:
            min_train_diff = abs(zero_train - target_train)
            min_test_diff = abs(zero_test - target_test)
            min_train_indices = train_indices
            min_test_indices = test_indices
        elif balance_focus == 'test_with_train_threshold' and abs(zero_test - target_test) < min_test_diff and train_balance < train_balance_threshold:
            min_train_diff = abs(zero_train - target_train)
            min_test_diff = abs(zero_test - target_test)
            min_train_indices = train_indices
            min_test_indices = test_indices
        elif balance_focus == 'train_with_test_threshold' and abs(zero_train - target_train) < min_train_diff and test_balance < test_balance_treshold:
            min_train_diff = abs(zero_train - target_train)
            min_test_diff = abs(zero_test - target_test)
            min_train_indices = train_indices
            min_test_indices = test_indices

    train_best, test_best = dataframe.iloc[min_train_indices].copy(), dataframe.iloc[min_test_indices].copy()

    return train_best, test_best

#____________________________________________________________________________________________________________________________________



