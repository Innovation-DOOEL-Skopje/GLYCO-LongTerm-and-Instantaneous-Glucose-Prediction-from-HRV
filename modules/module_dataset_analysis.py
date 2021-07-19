import pandas as pd
from typing import List, Dict, Union

#____________________________________________________________________________________________________________________________________


def quantitative_analysis(df: pd.DataFrame,
                          dataset_name: str,
                          class_feature: str,
                          classes: List[Union[str, int]],
                          ratios: bool = True,
                          overleaf: bool = False
                          ) -> Dict[str, Union[str, int, float]]:
    """
    Quantitative analysis of labeled dataset. Generate number of samples, percentages (if ratios == True) and balance (if binary class feature and ratios == True)

    :param df: the dataset in dataframe format
    :param dataset_name: descriptive name of the dataset
    :param class_feature:  name of class feature
    :param classes: list of class labels
    :param ratios: Whether to generate the ratio analysis

    :return:
    """

    # structures
    dataset_dict = dict()

    # initialize
    dataset_dict['dataset'] = dataset_name

    # SAMPLES ANALYSIS
    dataset_dict['Samples'] = df.shape[0]

    for class_iter in classes:

        dataset_dict[f'{class_iter} samples'] = (df[df[class_feature] == class_iter]).shape[0]

    # ratios include percentage of whole and ratio between classes if binary class feature
    if ratios:

        if not overleaf:

            for class_iter in classes:
                dataset_dict[f'{class_iter} samples (%)'] = round((df[df[class_feature] == class_iter]).shape[0] / df.shape[0], 4)

        if len(classes) == 2:
            dataset_dict[f'Balance of samples'] = round(dataset_dict[f'{classes[1]} samples'] / dataset_dict[f'{classes[0]} samples'], 4)

    # PATIENTS ANALYSIS
    dataset_dict['Patients'] = df['Patient_ID'].nunique()

    for class_iter in classes:

        dataset_dict[f'{class_iter} patients'] = (df[df[class_feature] == class_iter]['Patient_ID']).nunique()

    if ratios:

        if not overleaf:
            for class_iter in classes:
                dataset_dict[f'{class_iter} patients (%)'] = round((df[df[class_feature] == class_iter]['Patient_ID']).nunique() / df['Patient_ID'].nunique(), 4)

            if len(classes) == 2:
                dataset_dict[f'Balance patients'] = round(dataset_dict[f'{classes[1]} patients'] / dataset_dict[f'{classes[0]} patients'], 4)

    return dataset_dict















