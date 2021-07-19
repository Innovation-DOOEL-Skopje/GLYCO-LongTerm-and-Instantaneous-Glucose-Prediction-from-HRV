import sys
import pandas as pd
import numpy as np
import os
from typing import Union, List, Dict, Tuple


#____________________________________________________________________________________________________________________________________


def to_csv(df: Union[List[Dict], Dict, pd.DataFrame],
           path: str = None,
           save_index: bool = False
           ) -> None:
    """Save dataframe to .csv file without saving the index and with the potential of future customizations"""

    assert not path.endswith('.xlsx'), "Passing on xlsx file to csv writer"

    if not path.endswith('.csv'):
        path = path + '.csv'

    if type(df) is not pd.DataFrame:
        pd.DataFrame(df).to_csv(path, index = save_index)
    else:
        df.to_csv(path, index = save_index)

#____________________________________________________________________________________________________________________________________


def to_excel(df: Union[List[Dict], Dict, pd.DataFrame],
             path: str = None,
             save_index: bool = False
             ) -> None:
    """Save dataframe to .xlsx file without saving the index and with the potential of future customizations"""

    assert not path.endswith('.csv'), "Passing on csv file to xlsx writer"

    if not path.endswith('.xlsx'):
        path = path + '.xlsx'

    if type(df) is not pd.DataFrame:
        pd.DataFrame(df).to_excel(path, index = save_index)
    else:
        df.to_excel(path, index = save_index)

#____________________________________________________________________________________________________________________________________


def print_df(flexible_df: Union[List, pd.DataFrame]):

    if type(flexible_df) is list:
        flexible_df = pd.DataFrame(flexible_df)

    print(flexible_df.to_markdown())

#____________________________________________________________________________________________________________________________________


def save_numpy(data: Union[np.ndarray, List],
               path: str,
               warn_exists = True,
               ) -> None:
    """Save a numpy ndarray with validation check """

    if not path.endswith('.npy'):
        path = path + '.npy'

    if os.path.exists(path):
        if warn_exists:
            answer = str(input("""The numpy ndarray that you are trying to save already exists.
                                  Do you want to overwrite it? [y/n]"""))

            if answer.lower() == 'y': np.save(f'{path}', np.array(data))
            else: pass
























