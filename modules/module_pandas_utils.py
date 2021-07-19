from tabulate import tabulate
import pandas as pd


def print_df(df: pd.DataFrame):
    print(tabulate(df, headers = 'keys', tablefmt = 'psql'))