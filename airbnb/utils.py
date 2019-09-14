import pandas as pd
import numpy as np

def load_data(path):
    #loading all the files and checking their shapes respectively
    train_df = pd.read_csv(path+'train_users_2.csv',parse_dates=['date_account_created'])
    test_df = pd.read_csv(path+'test_users.csv',parse_dates=['date_account_created'])
    age_gender_df = pd.read_csv(path+'age_gender_bkts.csv')
    countries_df = pd.read_csv(path+'countries.csv')
    session_df = pd.read_csv(path+'sessions.csv')
    return train_df,test_df,age_gender_df,countries_df,session_df

#Training features extraction
def language(df):
    df['language'] = df['language'].apply(lambda x:'foreign' if x!='en' else x)
    return df


def browser(df):
    df['first_browser'] = df['first_browser'].apply(lambda x: "Mobile_Safari" if x=='Mobile Safari' else x)
    major_browser = ['Chrome','Safari','Firefox','IE','Mobile_Safari']
    df['first_browser'] = df['first_browser'].apply(lambda x : 'Other' if x not in major_browser else x)
    return df

def affiliate_provider(df):
    df['affiliate_provider'] = df['affiliate_provider'].apply(lambda x:'rest' if x not in
                                                              ['direct','google','other'] else x)
    return df


def handling_missng_values(df):
    df.loc[df.age > 120, 'age'] = np.nan
    df.age.fillna(df.age.mean(), inplace=True)
    df.drop(['date_first_booking', 'date_account_created', 'timestamp_first_active'], axis=1, inplace=True)
    return df

def training_feature(df):
    df = language(df)
    df = browser(df)
    df = affiliate_provider(df)
    df = handling_missng_values(df)
    return df




