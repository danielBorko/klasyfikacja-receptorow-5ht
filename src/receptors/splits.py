"""
Moduł podziału danych (Splits) z uwzględnieniem GroupKFold po gatunkach.
Zapewnia brak przecieku homologów między zbiorem treningowym a testowym.
"""

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from typing import Iterator, Tuple


def get_stratified_splits(metadata_df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> Iterator[
    Tuple[pd.Index, pd.Index]]:
    """
    Pomocniczy podział Stratified K-Fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = metadata_df.index
    y = metadata_df['label']
    return skf.split(X, y)


def get_group_splits(metadata_df: pd.DataFrame, n_splits: int = 5) -> Iterator[Tuple[pd.Index, pd.Index]]:
    """
    Główny podział GroupKFold oparty na kolumnie 'raw'.
    Gwarantuje, że sekwencje należące do tego samego gatunku nie trafią jednocześnie
    do treningu i testu w tym samym faldzie.
    """
    gkf = GroupKFold(n_splits=n_splits)
    X = metadata_df.index
    y = metadata_df['label']
    groups = metadata_df['raw']

    return gkf.split(X, y, groups)