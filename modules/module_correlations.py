import pandas as pd
from scipy import stats
import numpy as np
from typing import Union


# __________________________________________________________________________________________________________________

def pearson_nan_proof(x, y, round_flag: bool = False):

    two_feature_frame = pd.DataFrame({'x': x, 'y': y})
    where_both_not_null_frame = two_feature_frame[~two_feature_frame['x'].isna() & ~two_feature_frame['y'].isna()]
    if where_both_not_null_frame.shape[0] < 3:
        return np.nan
    else:
        return_val =  stats.pearsonr(where_both_not_null_frame['x'], where_both_not_null_frame['y'])[0]
        if round_flag: return round(return_val, 4)
        else: return return_val


def spearman_nan_proof(x, y, round_flag: bool = False):

    two_feature_frame = pd.DataFrame({'x': x, 'y': y})
    where_both_not_null_frame = two_feature_frame[~two_feature_frame['x'].isna() & ~two_feature_frame['y'].isna()]
    if where_both_not_null_frame.shape[0] < 3:
        return np.nan
    else:
        return_val = stats.spearmanr(where_both_not_null_frame['x'], where_both_not_null_frame['y'])[0]
        if round_flag: return round(return_val, 4)
        else: return return_val


def p_value_pearson_nan_proof(x, y, round_flag: bool = False):

    two_feature_frame = pd.DataFrame({'x': x, 'y': y})
    where_both_not_null_frame = two_feature_frame[~two_feature_frame['x'].isna() & ~two_feature_frame['y'].isna()]
    if where_both_not_null_frame.shape[0] < 3:
        return np.nan
    else:
        return_val =  stats.pearsonr(where_both_not_null_frame['x'], where_both_not_null_frame['y'])[1]
        if round_flag: return round(return_val, 4)
        else: return return_val


def p_value_spearman_nan_proof(x, y, round_flag: bool = False):

    two_feature_frame = pd.DataFrame({'x': x, 'y': y})
    where_both_not_null_frame = two_feature_frame[~two_feature_frame['x'].isna() & ~two_feature_frame['y'].isna()]
    if where_both_not_null_frame.shape[0] < 3:
        return np.nan
    else:
        return_val =  stats.spearmanr(where_both_not_null_frame['x'], where_both_not_null_frame['y'])[1]
        if round_flag: return round(return_val, 4)
        else: return return_val

# __________________________________________________________________________________________________________________


def target_correlations(df: pd.DataFrame, _method: str, ):
    # choose target feature
    if _method == 'PBC': target_feature = 'Class'
    else: target_feature = 'Glucose'

    # different correlation methods
    if _method == 'Spearman':
        cmtx = df.corr(method = 'spearman')
        pmtx = df.corr(method = p_value_spearman_nan_proof)
    else:
        cmtx = df.corr(method = 'pearson')
        pmtx = df.corr(method = p_value_pearson_nan_proof)

    return cmtx[target_feature], pmtx[target_feature]



