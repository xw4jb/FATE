import pandas as pd

FLOAT_ZERO = 1e-8

def get_empty_dataframe(df):
    return pd.DataFrame()

def hstack(df1, df2):
    return None

def sbt_hist(train_data, node_pos, g_tensor, h_tensor, 
             valid_features, use_missing, zero_as_missing):
    
    g_hist, h_hist, cnt, missing_g, missing_h = 1, 2, 3, 4, 5  # mock data, nevermind
    return g_hist, h_hist, cnt, missing_g, missing_h