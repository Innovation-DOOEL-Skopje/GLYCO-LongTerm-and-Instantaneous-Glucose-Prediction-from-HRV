import warnings
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
import pandas as pd
import numpy as np
from typing import List, Union, Tuple, Set

#____________________________________________________________________________________________________________________________________


def feature_engineering_end(df: pd.DataFrame,
                            feature_name: str = 'maen',
                            method: str = None
                            ) -> Union[pd.DataFrame, None]:
    """
    GLYCO-specific.
    Engineering the End feature which refers to the end of an ECG measurement for which the HRVs are calculated.

    @param df: Input dataframe.
    @param feature_name: The column name of the End feature.
    @param method: {'maen', 'cyclical'}
        The method to transform the end feature with.

    :return: The dataframe with transformed End feature.
    """

    # method can't be undefined
    if method is None:

        warnings.warn("Give the type of feature engineering as an argument")
        return None

    # morning, afternoon, evening, night - maen transformation
    if method == 'maen':

        df.replace(
            to_replace = {feature_name: {1: "night", 2: "night", 3: "night", 4: "night",
                                         5: "night", 6: "night", 7: "morning", 8: "morning",
                                         9: "morning", 10: "morning", 11: "morning", 12: "morning",
                                         13: "afternoon", 14: "afternoon", 15: "afternoon", 16: "afternoon",
                                         17: "afternoon", 18: "afternoon", 19: "evening", 20: "evening",
                                         21: "evening", 22: "evening", 23: "evening", 24: "evening", 0: "evening"}},
            inplace = True,
        )

    # cyclical transformation so there is a natural flow between 0h and 23h
    if method == 'cyclical':

        df[f'{feature_name}_sin'] = np.sin(df[feature_name] * (2. * np.pi / 24))
        df[f'{feature_name}_cos'] = np.cos(df[feature_name] * (2. * np.pi / 24))
        df.drop(feature_name, axis = 1, inplace = True)

    return df

#____________________________________________________________________________________________________________________________________


# class NullHandler:
#
#     def __init__(self, df, null):
#         pass
#
#
#     def initialize_nulls(df: pd.DataFrame, value_to_null: Union[int, float, str],
#                          features: Union[List[str], Set[str]]) -> pd.DataFrame:
#         """
#         @param df: input dataframe
#         @param value_to_null: value to replace with np.nan
#         @param features: features for which replacement is made
#
#         :return changed dataframe
#         """
#
#         features = list(features)
#
#         # initialize null values
#         df[features] = df[features].replace(to_replace = {value_to_null: np.nan})
#
#         return df
#
#
#     def remove_null_values(df: pd.DataFrame, features: Union[str, List[str], Set[str]] = None,
#                            return_num_removed: bool = False, ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int]]:
#         """
#         Remove rows with null values from a dataframe.
#
#         @param df: The input dataframe.
#         @param return_num_removed: Whether to return number of removed samples.
#
#         :return: The resulting dataframe
#         """
#
#         if features is None:
#             features = list(df.columns)
#         else:
#             features = list(features)
#
#         size_before = df.shape[0]
#
#         # remove rows with null values for the specified features
#         filter_df = df[features].copy()
#         df = df[~(filter_df.isna().any(axis = 1))]
#
#         if return_num_removed:
#             return df, size_before - df.shape[0]
#         else:
#             return df


def initialize_nulls(df: pd.DataFrame,
                     value_to_null: Union[int, float, str],
                     features: Union[List[str], Set[str]]
                     ) -> pd.DataFrame:
    """
    @param df: input dataframe
    @param value_to_null: value to replace with np.nan
    @param features: features for which replacement is made

    :return changed dataframe
    """

    features = list(features)

    # initialize null values
    df[features] = df[features].replace(to_replace = {value_to_null: np.nan})

    return df
#____________________________________________________________________________________________________________________________________


def remove_null_values(df: pd.DataFrame,
                       features: Union[str, List[str], Set[str]] = None,
                       return_num_removed: bool = False,
                       ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int]]:
    """
    Remove rows with null values from a dataframe.

    @param df: The input dataframe.
    @param return_num_removed: Whether to return number of removed samples.

    :return: The resulting dataframe
    """

    if features is None:
        features = list(df.columns)
    else:
        features = list(features)

    size_before = df.shape[0]

    # remove rows with null values for the specified features
    filter_df = df[features].copy()
    df = df[~(filter_df.isna().any(axis = 1))]

    if return_num_removed:
        return df, size_before - df.shape[0]
    else:
        return df

#____________________________________________________________________________________________________________________________________


def add_transformed_features(df: pd.DataFrame,
                             features_to_transform: Union[List[str], str] = None,
                             log: bool = False,
                             boxcox: bool = False,
                             yeojohnson: bool = False,
                             sqrt: bool = False):

    """ Add transformed features to a dataframe. Transformations include log, box-cox, square root...

    @param df: pandas.DataFrame
        Input dataframe
    @param features_to_transform: list of strings or string
        features to be transformed to the
    @param log: whether to append log transformed features
    @param sqrt: whether to append square root transformed features
    @param boxcox: whether to append box-cox transformed feature
    @param yeojohnson: whether to append yeo-johnson transformed feature

    :return: dataframe with appended transformed features
    """

    # defaults init
    if features_to_transform is None:
        features_to_transform = df.columns

    # cast to list, whether type is str or list
    features_to_transform = list(features_to_transform)

    for feature_name in features_to_transform:

        if log:

            # log(x+1) transform
            log_transformed = np.log1p(df[feature_name])
            df[feature_name + "_log"] = log_transformed

        if sqrt:

            # square root transform
            root_transformed = np.sqrt(df[feature_name])
            df[feature_name + "_root"] = root_transformed

        if boxcox:

            # box-cox power transform
            box_cox_transformed, _ = stats.boxcox(df[feature_name])
            df[feature_name + "_boxcox"] = box_cox_transformed

        if yeojohnson:

            # yeo-johnson power transform
            yeo_johnson_transformed, _ = stats.yeojohnson(df[feature_name])
            df[feature_name + "_yeojohnson"] = yeo_johnson_transformed

    return df

#____________________________________________________________________________________________________________________________________


def partial_polynomial_transform(df: pd.DataFrame,
                                 features_to_combine: Union[List[str], Set[str]] = None,
                                 degree=2):
    """
    Transform a dataframe such that all polynomial combinations of custom selected features of less than or equal
    custom degree are included in the dataframe.

    @param df: pandas.DataFrame
        Dataframe to be transformed
    @param features_to_combine: list of strings
        Features (list of column names) to be polynomialy expanded
    @param degree: int, default = 2
        Largest degree of the resulting polynomials

    :return: transformed dataframe
    """

    # defaults initialization
    if features_to_combine is None:
        features_to_combine = df.columns.to_list()
    else:
        # cast to list, whether set or list
        features_to_combine = list(features_to_combine)

    # holdout features in separate dataframe to be concated
    holdout_features = list(set(df.columns) - set(features_to_combine))
    holdout_dataframe = df[holdout_features].copy()

    # select subset to be polynomialy combined and handle null values
    prepolynomial_dataframe = df[features_to_combine].replace(np.nan, 0)

    # transform and create new dataframe
    polynomial_transformer = PolynomialFeatures(degree=degree)
    polynomial_ndarray = polynomial_transformer.fit_transform(prepolynomial_dataframe)
    polynomial_feature_names = polynomial_transformer.get_feature_names(input_features=features_to_combine)
    polynomial_dataframe = pd.DataFrame(data=polynomial_ndarray, columns=polynomial_feature_names)

    # reunite holdout and polynomial frames
    return pd.concat([holdout_dataframe.reset_index(drop = True), polynomial_dataframe], axis=1)

#____________________________________________________________________________________________________________________________________


def remove_outliers(df: pd.DataFrame,
                    method: str = 'iqr',
                    iqr_factor: float = 1.5,
                    features_to_analyze: Union[List[str], Set[str]] = None,
                    by_feature: bool = False,
                    with_nans: bool = False,
                    return_num_removed: bool = False,
                    plot_outlier_boundaries: bool = False,
                    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int]]:
    """
    Remove outliers from a dataframe.
    Outliers are considered data points which deviate more than 3 sigmas from the mean
    after computing the z-score.

    @param df: Dataframe from which outliers are removed.
    @param method: {'z-score', 'iqr'}, default = 'z-score' Method by which to detect the outliers
    @param features_to_analyze: cirteria features (list of column names) on which the outlier removal is based.
    If none is provided, transformation is performed on all features.
    @param with_nans: if True, null values won't influence removal of rows (used only with by_feature = False)
    @param by_feature: If True, outlier removal is performed on each column separately.
                        If False, if there is an outlier in one column, the whole row is removed.
    @param return_num_removed: Whether to return number of removed samples

    :return: DataFrame with removed outliers.
    """

    # if checking number of removed rows
    size_before_removal = df.shape[0]

    # defaults initialization
    if features_to_analyze is None:
        features_to_analyze = df.columns.to_list()
    else:
        # cast to list, whether set or list
        features_to_analyze = list(features_to_analyze)

    # holdout features in separate dataframe to be concated
    holdout_features = list(set(df.columns) - set(features_to_analyze))
    holdout_dataframe = df[holdout_features].copy()

    # outliers deviate more than 3 sigmas
    if method == 'z-score':

        # replacing outliers with np.nan
        if by_feature:

            # assigning to new df object
            df_without_outliers = pd.DataFrame()

            # replacing outliers with np.nan
            for feature_name_iter in features_to_analyze:
                feature = df[feature_name_iter]
                feature_zscore = (feature - feature.mean()) / feature.std()
                df_without_outliers[feature_name_iter] = feature.where(feature_zscore.abs() < 3)

            # reunite with features not analyzed for outliers
            df_without_outliers = pd.concat([holdout_dataframe, df_without_outliers], axis=1)

            # outcome info
            print('Number of rows removed: ', size_before_removal - df_without_outliers.shape[0])

            # return as specified
            if return_num_removed: return df_without_outliers, size_before_removal - df_without_outliers.shape[0]
            else: return df_without_outliers

        # removing rows
        else:

            # df on which to filter
            frame = df[features_to_analyze]

            # supporting dfs with nans - note: all rows with nans are removed
            if with_nans:

                frame_zscore = ((frame - frame.mean()) / frame.std(ddof = 0)).abs()

                # nans are replaced with zero ONLY IN FILTERING DATAFRAME, so that they don't influence the removal of rows
                frame_zscore.replace(to_replace = {np.nan: 0}, inplace = True)
                df_without_outliers = df[(frame_zscore <= 3).all(axis = 1)].reset_index(drop = True)

            # not supporting nans
            else:

                # filtering with scipy methods
                z_scores = stats.zscore(frame)
                abs_z_scores = np.abs(z_scores)
                filtered_entries = (abs_z_scores < 3).all(axis=1)
                df_without_outliers = df[filtered_entries]

            # outcome info
            print('Number of rows removed: ', size_before_removal - df_without_outliers.shape[0])

            # return as specified
            if return_num_removed: return df_without_outliers, size_before_removal - df_without_outliers.shape[0]
            else: return df_without_outliers

    # outliers deviate 1.5*IQR from 1st and 3rd quartile boundaries
    if method == 'iqr':

        if by_feature:
            df_without_outliers = pd.DataFrame()

            for feature in features_to_analyze:

                series = df[feature]

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = abs(q1 - q3)

                series_without_outliers_keep_index = series[~ ((series < (q1 - iqr_factor * iqr)) | (series > (q3 + iqr_factor * iqr)))]
                df_without_outliers = pd.concat([df_without_outliers, series_without_outliers_keep_index], axis = 1)

            for feature in list(set(df.columns) - set(features_to_analyze)):

                series = df[feature]
                df_without_outliers = pd.concat([df_without_outliers, series], axis = 1)

            if return_num_removed: return df_without_outliers, size_before_removal - df_without_outliers.shape[0]
            else: return df_without_outliers

        else:

            frame_to_filter = df[features_to_analyze]

            # calculate IQR
            q1 = frame_to_filter.quantile(0.25)
            q3 = frame_to_filter.quantile(0.75)
            iqr = (q1 - q3).abs()

            # remove rows which deviate more
            df_without_outliers = df[~ ((frame_to_filter < (q1 - iqr_factor*iqr)) | (frame_to_filter > (q3 + iqr_factor*iqr))).any(axis = 1)]

            if return_num_removed: return df_without_outliers, size_before_removal - df_without_outliers.shape[0]
            else: return df_without_outliers

#____________________________________________________________________________________________________________________________________


def power_transform(dataframe: pd.DataFrame,
                    power_transformer_method: str = 'yeo-johnson',
                    features_to_ignore: Union[str, List[str], Set[str]] = None,
                    ) -> pd.DataFrame:
    """
    Applies power transform to each feature of a dataframe.

    @param dataframe: The input dataframe.
    @param power_transformer_method: {'box-cox', 'yeo-johnson'}
        The power transform method.
    @param features_to_ignore: Features to ignore when applying power transformations like string features


    :return: the transformed dataframe.
    """

    # count of non-null values per feature
    count_of_non_null = dataframe.count()

    # dataframe to append when working with features with 0, 1, 2 non-null values for consistency
    # later is removed so it has no effect on the transformation
    append_df = pd.DataFrame()

    if features_to_ignore is not None:

        # cast to list for iteration and select subset of df
        features_to_ignore = list(features_to_ignore)
        holdout_df = dataframe[features_to_ignore].copy()

        # transformed features have new indices, so old need to be reset as well
        holdout_df.reset_index(inplace = True, drop = True)

        # drop them from df on which transforms are applied
        dataframe.drop(features_to_ignore, axis = 1, inplace = True)

    # creating 3 more rows to append for consistency
    for column_iter in dataframe.columns:

        # if count_of_non_null[column_iter] in [0, 1, 2]:
        #     raise ValueError(f'Feature {column_iter} has only 1 or 2 non-null values. Handle those appropriately before passing the dataframe to the mehtod.')

        append_df[column_iter] = [0, 0, 0] if count_of_non_null[column_iter] in [0, 1, 2] else [np.nan for _iter in range(3)]

    # appending the rows
    dataframe_expanded = dataframe.append(append_df, ignore_index = True)

    # applying the transformation
    power_transformer = PowerTransformer(method = power_transformer_method, standardize=False)
    dataframe_transformed = pd.DataFrame(data = power_transformer.fit_transform(dataframe_expanded),
                                         columns = dataframe_expanded.columns)

    # dropping the appended rows
    dataframe_transformed = dataframe_transformed.drop(dataframe_transformed.tail(3).index)

    # if features were ignored, append them before returning
    if features_to_ignore is not None:
        dataframe_transformed = pd.concat([dataframe_transformed, holdout_df], axis = 1)

    return dataframe_transformed

#____________________________________________________________________________________________________________________________________
