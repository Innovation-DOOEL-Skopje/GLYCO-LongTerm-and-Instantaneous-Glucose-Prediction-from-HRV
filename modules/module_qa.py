import pandas as pd


def print_df_info(df: pd.DataFrame,
                  print_columns: bool = False,
                  print_info: bool = True,
                  print_desc_stats = False
                  ) -> None:

    """
    @param df: dataset dataframe for which analysis is printed
               to make sure there are no mistakes in (pre)processing
    @ print_columns:
    @ print_info:
    @ print_desc_status:
    """

    print('_________________________________________________________')
    if print_columns:
        print('Dataset columns:')
        print(df.columns)
        print('\n\n')
    if print_info:
        print('Non-null counts:')
        print(df.info())
    if print_desc_stats:
        print('\n\n')
        print('Description')
        print(df.describe())
    print('_________________________________________________________')
    print('\n\n')
    print('\n\n')
    print('\n\n')



