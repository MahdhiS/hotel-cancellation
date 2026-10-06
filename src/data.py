import pandas as pd

def load_basic_clean(df):
    df_duplicates_dropped = _drop_duplicates(df)
    df_missing_dropped = _drop_dupl_contry_children(df_duplicates_dropped)
    df_missing_filled = _fill_missing_agent_company(df_missing_dropped)
    return df_missing_filled


def _drop_duplicates(df):
    return df.drop_duplicates()


def _drop_dupl_contry_children(df):
    return df.dropna(subset=['country', 'children'])


def _fill_missing_agent_company(df):
    return df.fillna(value=0)