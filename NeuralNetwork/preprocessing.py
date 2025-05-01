import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pgeocode
from sklearn.model_selection import train_test_split


def epcToNumeric(row):
    region = row['region']
    epc_score = row['epcScore']
    
    epc_mapping = {
        'Flanders': {
            'A++': 0,
            'A+': 0,
            'A': 100,
            'B': 200,
            'C': 300,
            'D': 400,
            'E': 500,
            'F': 600,
            'G': 700
        },
        'Wallonia': {
            'A++': 0,
            'A+': 50,
            'A': 90,
            'B': 170,
            'C': 250,
            'D': 330,
            'E': 420,
            'F': 510,
            'G': 600
        },
        'Bruxelles': {
            'A++': 0,
            'A+': 0,
            'A': 45,
            'B': 95,
            'C': 145,
            'D': 210,
            'E': 275,
            'F': 345,
            'G': 450
        }
    }
    
    return epc_mapping.get(region, {}).get(epc_score, None)

def pricePerM2(df):
    df['pricePerM2'] = df['price']/df['habitableSurface']
    return df

def getCoordinates(df):
    nomi = pgeocode.Nominatim('be')
    df['postCode'] = df['postCode'].astype(str)
    unique_postcodes = df["postCode"].astype(str).unique()

    geo_df = nomi.query_postal_code(list(unique_postcodes))

    geo_df = geo_df[['postal_code', 'latitude', 'longitude']]
    geo_df = geo_df.rename(columns={'postal_code': 'postCode'})

    geo_df['postCode'] = geo_df['postCode'].astype(str)

    df = df.merge(geo_df, on='postCode', how='left')
    #df['postCode'] = df['postCode'].astype(int)
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

# Make a cleaning function :

def transform_data_types(df, col_types):
    for col, dtype in col_types.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df

def cleaning(df):
    df = df.drop(columns=["Unnamed: 0", "url"])

    df = df.drop(columns=['monthlyCost', 'hasBalcony', 'accessibleDisabledPeople', 'roomCount', 'diningRoomSurface', 
                          'streetFacadeWidth', 'gardenOrientation', 'kitchenSurface', 'floorCount', 'hasDiningRoom', 
                          'hasDressingRoom'])
    
    
    binary_cols = [
        'hasBasement', 'hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 
        'hasAirConditioning', 'hasArmoredDoor', 'hasVisiophone', 'hasOffice', 
        'hasSwimmingPool', 'hasFireplace', 'parkingCountIndoor', 'parkingCountOutdoor',
        'hasAttic'
    ]
    
    for col in binary_cols:
        df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
    
    # Colonnes dépendantes d'autres colonnes
    df['hasLivingRoom'] = df['hasLivingRoom'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    df.loc[df['hasLivingRoom'].isna(), 'hasLivingRoom'] = df['livingRoomSurface'].notnull().astype(int)
    
    df['hasGarden'] = df['hasGarden'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    df.loc[df['hasGarden'].isna(), 'hasGarden'] = df['gardenSurface'].notnull().astype(int)
    
    df['hasTerrace'] = df['hasTerrace'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    df.loc[df['hasTerrace'].isna(), 'hasTerrace'] = df['terraceSurface'].notnull().astype(int)
    
    # When hasLivingRoom = 0 ; livingRoomSurface = 0
    df.loc[df['hasLivingRoom'] == 0, 'livingRoomSurface'] = 0
    
    # When hasGarden = 0 ; gardenSurface = 0
    df.loc[df['hasGarden'] == 0, 'gardenSurface'] = 0
    
    # When hasTerrace = 0 ; terraceSurface = 0 and terraceOrientation = 0
    df.loc[df['hasTerrace'] == 0, 'terraceSurface'] = 0
    df.loc[df['hasTerrace'] == 0, 'terraceOrientation'] = 0
    
    #drop number of facade bigger than 4 and transform "facedeCount" into "facadeCount"
    df['facadeCount'] = df['facedeCount']
    df = df.drop(columns='facedeCount')
    df['facadeCount'] = df['facadeCount'].fillna(2)
    '''df = df[df['facadeCount'] <= 4]'''
    
    # bedroomCount : lets assume that they have at least one so fill nan by 1
    df['bedroomCount'] = df['bedroomCount'].fillna(1).astype(float)
    
    # bathroomCount same as bedrooms
    df['bathroomCount'] = df['bathroomCount'].fillna(1).astype(float)
    
    # toiletCount same as bedrooms
    df['toiletCount'] = df['toiletCount'].fillna(1).astype(float)
    
    # habitableSurface : replace by median 
    #df['habitableSurface'] = df['habitableSurface'].fillna(df['habitableSurface'].median())
    mediane_by_subtype = df.groupby('subtype')['habitableSurface'].median()
    df['habitableSurface'] = df.apply(
        lambda row: mediane_by_subtype[row['subtype']] if pd.isna(row['habitableSurface']) else row['habitableSurface'],
        axis=1
    )
    
    # buildingCondition : replace by 'NOT_MENTIONED
    df['buildingCondition'] = df['buildingCondition'].fillna('NOT_MENTIONED')
    
    # buildingConstructionYear
    df['buildingConstructionYear'] = df['buildingConstructionYear'].fillna(df['buildingConstructionYear'].median()).astype(int)
    
    
    # floodZoneType lts assume that missing values are NON_FLOOD_ZONE
    df['floodZoneType'] = df['floodZoneType'].fillna('NON_FLOOD_ZONE')
    
    # heatingType
    df['heatingType'] = df['heatingType'].fillna(df['heatingType'].mode()[0])
    
    # hasThermicPanels lets assume that if its not precised, there are not
    df['hasThermicPanels'] = df['hasThermicPanels'].fillna(0).astype(float)
    
    # kitchenType
    df['kitchenType'] = df['kitchenType'].fillna(df['kitchenType'].mode()[0])
    
    # landSurface
    df['landSurface'] = df['landSurface'].fillna(df['landSurface'].median())
    
    # livingRoomSurface
    df['livingRoomSurface'] = df['livingRoomSurface'].fillna(df['livingRoomSurface'].median())
    
    # terraceSurface
    median_terrace = df.loc[(df['hasTerrace'] == 1) & (df['terraceSurface'].notnull()), 'terraceSurface'].median()
    df.loc[(df['hasTerrace'] == 1) & (df['terraceSurface'].isna()), 'terraceSurface'] = median_terrace
    df.loc[(df['hasTerrace'] != 1) & (df['terraceSurface'].isna()), 'terraceSurface'] = 0
    
    # terraceOrientation
    mode_terrace = df.loc[(df['hasTerrace'] == 1), 'terraceOrientation'].mode()[0]
    df.loc[(df['hasTerrace'] == 1) & (df['terraceOrientation'].isna()), 'terraceOrientation'] = mode_terrace
    df.loc[(df['hasTerrace'] != 1) & (df['terraceOrientation'].isna()), 'terraceOrientation'] = 'NO_TERRACE'

    
    col_types = {'id': 'int', 'type': 'str', 'subtype': 'str', 'bedroomCount': 'int', 'bathroomCount': 'int',
                 'province': 'str', 'locality': 'str', 'postCode': 'int', 'habitableSurface': 'float', 
                 'hasBasement': 'int', 'buildingCondition': 'str',
                 'buildingConstructionYear': 'int', 'hasLift': 'int', 'floodZoneType': 'str',
                 'heatingType': 'str', 'hasHeatPump': 'int', 'hasPhotovoltaicPanels': 'int', 'hasThermicPanels': 'int',
                 'kitchenType': 'str', 'landSurface': 'float', 'hasLivingRoom': 'int', 'livingRoomSurface': 'float',
                 'hasGarden': 'int', 'gardenSurface': 'float', 'parkingCountIndoor': 'int', 'parkingCountOutdoor': 'int',
                 'hasAirConditioning': 'int', 'hasArmoredDoor': 'int', 'hasVisiophone': 'int', 'hasOffice': 'int', 
                 'toiletCount': 'int', 'hasSwimmingPool': 'int', 'hasFireplace': 'int', 'hasTerrace': 'int', 'terraceSurface': 'float',
                 'terraceOrientation': 'str', 'epcScore': 'str', 'facadeCount': 'int'}
    
    df = transform_data_types(df, col_types)
###
###
###
    # Type into isHouse -> if false : Apartment
    df['isHouse'] = (df['type'] == 'HOUSE').astype(int)

    # subtype -> in pipeline

    # province ? drop or dummies ?
    df = pd.get_dummies(df, columns=['province'], prefix='province', dtype=int)
    
    # locality ? drop because zipcode

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

    # floodzone type 
    df['floodZoneType'] = (df['floodZoneType'] != 'NON_FLOOD_ZONE').astype(int)
    
    # heatingType
    df = pd.get_dummies(df, columns=['heatingType'], prefix='heating', dtype=int)
    
    # kitchenType
    df = pd.get_dummies(df, columns=['kitchenType'], prefix='kitchen', dtype=int)

    # add region information
    def get_region(zip_code):
        if 1000 <= zip_code <= 1299:
            return "Bruxelles"
        elif 1300 <= zip_code <= 1499 or 4000 <= zip_code <= 7999:
            return "Wallonia"
        else:
            return "Flanders"
    
    df['region'] = df['postCode'].apply(get_region)

    # postCode
    #df['postCode'] = df['postCode'].astype(int)

    # epcScore
    df['epcScore'] = df.apply(epcToNumeric, axis=1)

    df = pricePerM2(df)
    df = getCoordinatesGiraffe(df)
    
    df = df.dropna(subset=['latitude', 'longitude'])

    df = df.drop(columns=['type', 'locality', 'region'])
    
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
            # Si le prix courant n'existe pas (ex : en test), on évalue à la moyenne des voisins
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
    #df = df[(df['price']<2000000) & (df['price']>100000)]
    df = df[(df['price']<1000000)]
    # drop lines without price
    df = df.dropna(subset=['price'])
    # epcScore
    epc_order = ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    df = df[df['epcScore'].isin(epc_order)]
    df['epcScore'] = df['epcScore'].fillna(df['epcScore'].mode()[0])
    df = transform_data_types(df, {'price': float})
    return df

def trainTestClean():

    df = preprocess()

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    df_train = cleaning(df_train)
    df_test = cleaning(df_test)

    df_train = pd.get_dummies(df_train, columns=['subtype', 'terraceOrientation'], dtype=int)
    df_test = pd.get_dummies(df_test, columns=['subtype', 'terraceOrientation'], dtype=int)

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

    df_scaled['buildingAge'] = 2025 - df_scaled['buildingConstructionYear']

    binary_cols = [col for col in df.columns if set(df[col].unique()) <= {0, 1}]

    exclude_cols = set(binary_cols + ["id", "price"])

    continuous_cols = [col for col in df.columns if col not in exclude_cols]

    scaler = StandardScaler()
    df_scaled[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    return df_scaled

def main():
    df_train, df_test = trainTestClean()
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
