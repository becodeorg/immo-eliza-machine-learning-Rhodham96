import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pgeocode
from sklearn.model_selection import train_test_split


def col_str_to_int(df, col):
    return pd.factorize(df[col])[0] + 1

def clean_factorize():
    
    df = pd.read_csv('./data/Kangaroo.csv')
    df = df.drop(columns=["Unnamed: 0", "url"])

    df = df.drop(columns=[
        'id', 'monthlyCost', 'hasBalcony', 'accessibleDisabledPeople', 'roomCount', 
        'diningRoomSurface', 'streetFacadeWidth', 
        'kitchenSurface', 'floorCount', 'hasDiningRoom', 'hasDressingRoom'
    ])

    str_cols = [
        'locality', 'terraceOrientation', 'epcScore', 'gardenOrientation',
        'kitchenType', 'heatingType', 'floodZoneType', 'buildingCondition',
        'type', 'subtype', 'province'
    ]

    for col in str_cols:
        df[col] = col_str_to_int(df, col)

    df = df.replace({True: 1, False: 0})
    df = df.fillna(-1)

    return df

def trainTestClean():

    df = clean_factorize()
    df = df[(df['price']<1000000)]
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    return df_train, df_test


#if __name__ == "__main__":
#    main()
