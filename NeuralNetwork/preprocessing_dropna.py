import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pgeocode
from sklearn.model_selection import train_test_split
    

def pricePerM2(df):
    df['pricePerM2'] = df['price']/df['habitableSurface']
    return df

def getCoordinatesGiraffe(df):
    df_giraffe = pd.read_csv('../data/Giraffe.csv')
    df_giraffe = df_giraffe[['propertyId', 'latitude', 'longitude']]

    df_giraffe['id'] = df_giraffe['propertyId']
    cols = df_giraffe.columns.tolist()
    cols.remove('id')
    new_order = ['id'] + cols
    df_giraffe = df_giraffe[new_order]

    df_giraffe = df_giraffe.drop(columns='propertyId')

    df = df.merge(df_giraffe, on='id', how='left')
    df = df.dropna(subset=['latitude', 'longitude'])
    return df

def cleaning(df):

    df = df.drop(columns=["Unnamed: 0", "url"])

    df = df.drop(columns=['monthlyCost', 'hasBalcony', 'accessibleDisabledPeople', 'roomCount', 'diningRoomSurface', 
                          'streetFacadeWidth', 'gardenOrientation', 'kitchenSurface', 'floorCount', 'hasDiningRoom', 
                          'hasDressingRoom'])
    
    df = df.dropna(subset=['bedroomCount', 'habitableSurface'])
    print(df.shape)
    binary_cols = [
        'hasBasement', 'hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 
        'hasAirConditioning', 'hasArmoredDoor', 'hasVisiophone', 'hasOffice', 
        'hasSwimmingPool', 'hasFireplace', 'parkingCountIndoor', 'parkingCountOutdoor',
        'hasAttic', 'hasLivingRoom', 'hasGarden', 'hasTerrace', 'hasLivingRoom', 
    ]
    
    for col in binary_cols:
        df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(-1).astype(int)
    
    df['buildingCondition'] = df['buildingCondition'].fillna('NOT_MENTIONED')

    df['buildingAge'] = 2025 - df['buildingConstructionYear']
    df['buildingConstructionYear'] = df['buildingConstructionYear'].fillna(-1).astype(int)

    df['isHouse'] = (df['type'] == 'HOUSE').astype(int)

    # floodZoneType lts assume that missing values are NON_FLOOD_ZONE
    df['floodZoneType'] = df['floodZoneType'].fillna('NON_FLOOD_ZONE')
    df['floodZoneType'] = (df['floodZoneType'] != 'NON_FLOOD_ZONE').astype(int)

    # building condition 
    condition_rating = {
        'to restore': 0,
        'to renovate': 1,
        'to be done up': 2,
        'good': 3,
        'just renovated': 4,
        'as new': 5
    }
    df['buildingCondition'] = (df['buildingCondition'].astype(str).str.strip().str.lower()
                                    .map(condition_rating).fillna(-1).astype(int))
    epc_mapping = {
        'A+': 1,
        'A': 2,
        'B': 3,
        'C': 4,
        'D': 5,
        'E': 6,
        'F': 7,
        'G': 8
    }

    df['epcScore'] = df['epcScore'].map(epc_mapping)

    df = pricePerM2(df)
    df = getCoordinatesGiraffe(df)
    df = df.dropna(subset=['latitude', 'longitude'])

    # getdummies
    df = pd.get_dummies(df, columns=['province', 'buildingCondition', 'heatingType', 
                                     'kitchenType','subtype', 'terraceOrientation'], dtype=int)
    
    df = df.dropna(subset=['latitude', 'longitude'])

    df = df.drop(columns=['type', 'locality', 'buildingConstructionYear'])

    # FILLNA
    df = df.fillna(-1).astype(float)
    return df


def fit_kde_knn_model(df_train, k=20):
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(df_train[['latitude', 'longitude']])

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(coords_scaled)

    return scaler, knn

def apply_kde_knn(df, df_train, scaler, knn):
    coords_scaled = scaler.transform(df[['latitude', 'longitude']])
    distances, indices = knn.kneighbors(coords_scaled)

    kde_scores = []
    has_kde_score = []

    for i in range(len(df)):
        neighbor_idxs = indices[i]
        neighbor_prices = df_train['pricePerM2'].iloc[neighbor_idxs].dropna()

        if len(neighbor_prices) < 2:
            kde_scores.append(np.nan)
            has_kde_score.append(0)
        else:
            kde = gaussian_kde(neighbor_prices)
            value_to_evaluate = df['pricePerM2'].iloc[i] if 'pricePerM2' in df.columns else neighbor_prices.mean()
            kde_scores.append(kde(value_to_evaluate)[0])
            has_kde_score.append(1)

    df['kde_price_per_m2_knn'] = kde_scores
    df['has_kde_score'] = has_kde_score
    df['kde_price_per_m2_knn'] = df['kde_price_per_m2_knn'].fillna(np.nanmedian(kde_scores))

    return df.drop(columns=['latitude', 'longitude'], errors='ignore')


def preprocess():
    df = pd.read_csv("../data/Kangaroo.csv")
    df = df.drop_duplicates(subset=["id"], keep="first")
    df = df[(df['price']<1500000)]
    # drop lines without price
    df = df.dropna(subset=['price'])
    # epcScore
    epc_order = ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    df = df[df['epcScore'].isin(epc_order)]
    df['epcScore'] = df['epcScore'].fillna(df['epcScore'].mode()[0])
    df['price'] = df['price'].astype(float)
    return df

def trainTestCleanDropNa():

    df = preprocess()

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    df_train = cleaning(df_train)
    df_test = cleaning(df_test)

    scaler, knn = fit_kde_knn_model(df_train)
    df_train = apply_kde_knn(df_train, df_train, scaler, knn)
    df_test = apply_kde_knn(df_test, df_train, scaler, knn)

    df_train['postCode'] = df_train['postCode'].astype(int)
    df_test['postCode'] = df_test['postCode'].astype(int)
    
    df_train = reorganizeColumns(df_train)
    df_test = reorganizeColumns(df_test)

    return df_train, df_test

def reorganizeColumns(df):
    cols = df.columns.tolist()
    cols.remove('id')
    cols.remove('price')
    new_order = ['id'] + cols + ['price']
    df = df[new_order]
    return df

def standardize(df):
    df_scaled = df.copy()

    binary_cols = [col for col in df.columns if set(df[col].unique()) <= {0, 1}]

    exclude_cols = set(binary_cols + ["id", "price"])

    continuous_cols = [col for col in df.columns if col not in exclude_cols]

    scaler = StandardScaler()
    df_scaled[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    return df_scaled

def main():
    df_train, df_test = trainTestCleanDropNa()
    df_train = standardize(df_train)
    df_test = standardize(df_test)
    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()


    print("df_train columns")
    print(df_train.columns)
    print("df_test columns")
    print(df_test.columns)
    print(df_test.columns)
    print('postCode' in df_test.columns)  # Should print True
    print('buildingConstructionYear' in df_test.columns)  # Should print True


    df_train = df_train.drop(columns=['id', 'postCode','buildingConstructionYear'])
    df_test = df_test.drop(columns=['id', 'postCode','buildingConstructionYear'])

    df_train = reorganizeColumns(df_train)
    df_test = reorganizeColumns(df_test)

    df_train.to_csv("../data/train.csv", index=False)
    df_test.to_csv("../data/test.csv", index=False)


if __name__ == "__main__":
    main()
