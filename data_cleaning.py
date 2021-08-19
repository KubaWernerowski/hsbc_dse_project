import numpy as np
import pandas as pd


# ----------- #
# binary data #
# ----------- #
def clean_binary_cols(df):
    '''
    Changes the following columns to 0 or 1.
    - default
    - housing
    - loan
    - y
    '''
    mapping = {'no': 0, 'yes': 1}
    replace = {
        'default': mapping,
        'housing': mapping,
        'loan': mapping,
        'y': mapping
    }
    
    df = df.replace(replace)
    
    return df

def was_previously_contacted(pdays_col):
    if pdays_col == -1:
        return 0
    return 1


# --------------------- #
# functions for months  #
# --------------------- #

def month_to_quarters(month_col):
    Q1 = ['jan', 'feb', 'mar']
    Q2 = ['apr', 'may', 'june']
    Q3 = ['jul', 'aug', 'sep']
    Q4 = ['oct', 'nov', 'dec']
    
    if month_col in Q1:
        return 'Q1'
    elif month_col in Q2:
        return 'Q2'
    elif month_col in Q3:
        return 'Q3'
    else:
        return 'Q4'


def month_to_numeric(month_col, make_jan_eq_0=False):
    mapping = {'jan': 1, 'feb': 2, 'mar': 3,
               'apr': 4, 'may': 5, 'jun': 6,
               'jul': 7, 'aug': 8, 'sep': 9,
               'oct': 10, 'nov': 11, 'dec': 12}
    
    if make_jan_eq_0:
        for key in mapping:
            mapping[key] -= 1
    
    return mapping[month_col]


# ------- #
# dummies #
# ------- #

def get_cat_cols_dummies(df, drop_original_cols=False, drop_first=False):
    '''
    Adds dummy variables for columns.
    Applies to:
    - job
    - marital
    - education
    - contact
    - poutcome
    '''
    col_names = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    dummies = pd.get_dummies(df, columns=col_names, drop_first=drop_first)
    
    if drop_original_cols:
        df = df.drop(col_names, axis=1)
        
    df = pd.concat([df, dummies], axis=1)
    
    return df
