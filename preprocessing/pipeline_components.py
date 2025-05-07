import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pgeocode

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.col_types = {
            'id': 'int', 'type': 'str', 'subtype': 'str', 'bedroomCount': 'int', 
            'bathroomCount': 'int', 'province': 'str', 'locality': 'str', 
            'postCode': 'int', 'habitableSurface': 'float', 'hasBasement': 'int', 
            'buildingCondition': 'str', 'buildingConstructionYear': 'int', 
            'hasLift': 'int', 'floodZoneType': 'str', 'heatingType': 'str', 
            'hasHeatPump': 'int', 'hasPhotovoltaicPanels': 'int', 
            'hasThermicPanels': 'int', 'kitchenType': 'str', 'landSurface': 'float', 
            'hasLivingRoom': 'int', 'livingRoomSurface': 'float', 'hasGarden': 'int', 
            'gardenSurface': 'float', 'parkingCountIndoor': 'int', 
            'parkingCountOutdoor': 'int', 'hasAirConditioning': 'int', 
            'hasArmoredDoor': 'int', 'hasVisiophone': 'int', 'hasOffice': 'int', 
            'toiletCount': 'int', 'hasSwimmingPool': 'int', 'hasFireplace': 'int', 
            'hasTerrace': 'int', 'terraceSurface': 'float', 'terraceOrientation': 'str', 
            'epcScore': 'str', 'facadeCount': 'int'
        }
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # Drop unnecessary columns
        df = df.drop(columns=["Unnamed: 0", "url"])
        df = df.drop(columns=['monthlyCost', 'hasBalcony', 'accessibleDisabledPeople', 
                            'roomCount', 'diningRoomSurface', 'streetFacadeWidth', 
                            'gardenOrientation', 'kitchenSurface', 'floorCount', 
                            'hasDiningRoom', 'hasDressingRoom'])
        
        # Handle binary columns
        binary_cols = [
            'hasBasement', 'hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 
            'hasAirConditioning', 'hasArmoredDoor', 'hasVisiophone', 'hasOffice', 
            'hasSwimmingPool', 'hasFireplace', 'parkingCountIndoor', 'parkingCountOutdoor',
            'hasAttic'
        ]
        
        for col in binary_cols:
            df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
        
        # Handle dependent columns
        df['hasLivingRoom'] = df['hasLivingRoom'].map({True: 1, False: 0, 'True': 1, 'False': 0})
        df.loc[df['hasLivingRoom'].isna(), 'hasLivingRoom'] = df['livingRoomSurface'].notnull().astype(int)
        
        df['hasGarden'] = df['hasGarden'].map({True: 1, False: 0, 'True': 1, 'False': 0})
        df.loc[df['hasGarden'].isna(), 'hasGarden'] = df['gardenSurface'].notnull().astype(int)
        
        df['hasTerrace'] = df['hasTerrace'].map({True: 1, False: 0, 'True': 1, 'False': 0})
        df.loc[df['hasTerrace'].isna(), 'hasTerrace'] = df['terraceSurface'].notnull().astype(int)
        
        # Set surfaces to 0 when feature is not present
        df.loc[df['hasLivingRoom'] == 0, 'livingRoomSurface'] = 0
        df.loc[df['hasGarden'] == 0, 'gardenSurface'] = 0
        df.loc[df['hasTerrace'] == 0, 'terraceSurface'] = 0
        df.loc[df['hasTerrace'] == 0, 'terraceOrientation'] = 0
        
        # Handle facade count
        df['facadeCount'] = df['facedeCount']
        df = df.drop(columns='facedeCount')
        df['facadeCount'] = df['facadeCount'].fillna(2)
        
        # Fill missing values
        df['bedroomCount'] = df['bedroomCount'].fillna(1).astype(float)
        df['bathroomCount'] = df['bathroomCount'].fillna(1).astype(float)
        df['toiletCount'] = df['toiletCount'].fillna(1).astype(float)
        
        # Fill habitable surface with median by subtype
        mediane_by_subtype = df.groupby('subtype')['habitableSurface'].median()
        df['habitableSurface'] = df.apply(
            lambda row: mediane_by_subtype[row['subtype']] if pd.isna(row['habitableSurface']) else row['habitableSurface'],
            axis=1
        )
        
        # Fill other missing values
        df['buildingCondition'] = df['buildingCondition'].fillna('NOT_MENTIONED')
        df['buildingConstructionYear'] = df['buildingConstructionYear'].fillna(df['buildingConstructionYear'].median()).astype(int)
        df['floodZoneType'] = df['floodZoneType'].fillna('NON_FLOOD_ZONE')
        df['heatingType'] = df['heatingType'].fillna(df['heatingType'].mode()[0])
        df['hasThermicPanels'] = df['hasThermicPanels'].fillna(0).astype(float)
        df['kitchenType'] = df['kitchenType'].fillna(df['kitchenType'].mode()[0])
        df['landSurface'] = df['landSurface'].fillna(df['landSurface'].median())
        df['livingRoomSurface'] = df['livingRoomSurface'].fillna(df['livingRoomSurface'].median())
        
        # Handle terrace surface and orientation
        median_terrace = df.loc[(df['hasTerrace'] == 1) & (df['terraceSurface'].notnull()), 'terraceSurface'].median()
        df.loc[(df['hasTerrace'] == 1) & (df['terraceSurface'].isna()), 'terraceSurface'] = median_terrace
        df.loc[(df['hasTerrace'] != 1) & (df['terraceSurface'].isna()), 'terraceSurface'] = 0
        
        mode_terrace = df.loc[(df['hasTerrace'] == 1), 'terraceOrientation'].mode()[0]
        df.loc[(df['hasTerrace'] == 1) & (df['terraceOrientation'].isna()), 'terraceOrientation'] = mode_terrace
        df.loc[(df['hasTerrace'] != 1) & (df['terraceOrientation'].isna()), 'terraceOrientation'] = 'NO_TERRACE'
        
        # Convert data types
        for col, dtype in self.col_types.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        
        return df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.epc_mapping = {
            'Flanders': {
                'A++': 0, 'A+': 0, 'A': 100, 'B': 200, 'C': 300,
                'D': 400, 'E': 500, 'F': 600, 'G': 700
            },
            'Wallonia': {
                'A++': 0, 'A+': 50, 'A': 90, 'B': 170, 'C': 250,
                'D': 330, 'E': 420, 'F': 510, 'G': 600
            },
            'Bruxelles': {
                'A++': 0, 'A+': 0, 'A': 45, 'B': 95, 'C': 145,
                'D': 210, 'E': 275, 'F': 345, 'G': 450
            }
        }
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # Log initial state
        print(f"Initial data shape: {df.shape}")
        print(f"Price range: [{df['price'].min()}, {df['price'].max()}]")
        print(f"HabitableSurface range: [{df['habitableSurface'].min()}, {df['habitableSurface'].max()}]")
        
        # Filter out extremely high prices
        high_price_count = (df['price'] > 1500000).sum()
        print(f"Found {high_price_count} properties with price above 1,500,000")
        df = df[df['price'] <= 1500000]
        print(f"Data shape after price filter: {df.shape}")
        
        # Check for problematic values
        zero_price = (df['price'] <= 0).sum()
        zero_surface = (df['habitableSurface'] <= 0).sum()
        print(f"Found {zero_price} zero/negative prices and {zero_surface} zero/negative surfaces")
        
        # Handle problematic values
        if zero_price > 0:
            print("Replacing zero/negative prices with NaN")
            df.loc[df['price'] <= 0, 'price'] = np.nan
            
        if zero_surface > 0:
            print("Replacing zero/negative surfaces with NaN")
            df.loc[df['habitableSurface'] <= 0, 'habitableSurface'] = np.nan
        
        # Add isHouse feature
        df['isHouse'] = (df['type'] == 'HOUSE').astype(int)
        
        # Add region information first
        def get_region(zip_code):
            if 1000 <= zip_code <= 1299:
                return "Bruxelles"
            elif 1300 <= zip_code <= 1499 or 4000 <= zip_code <= 7999:
                return "Wallonia"
            else:
                return "Flanders"
        
        df['region'] = df['postCode'].apply(get_region)
        
        # Now add price per m2
        df['pricePerM2'] = df['price'] / df['habitableSurface']
        
        # Log price per m2 statistics
        print(f"Price per m2 range: [{df['pricePerM2'].min()}, {df['pricePerM2'].max()}]")
        print(f"Number of inf values in price per m2: {(df['pricePerM2'] == np.inf).sum()}")
        print(f"Number of -inf values in price per m2: {(df['pricePerM2'] == -np.inf).sum()}")
        print(f"Number of NaN values in price per m2: {df['pricePerM2'].isna().sum()}")
        
        # Handle inf values
        df['pricePerM2'] = df['pricePerM2'].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median by region
        median_by_region = df.groupby('region')['pricePerM2'].median()
        df['pricePerM2'] = df.apply(
            lambda row: median_by_region[row['region']] if pd.isna(row['pricePerM2']) else row['pricePerM2'],
            axis=1
        )
        
        # Convert EPC score
        df['epcScore'] = df.apply(lambda row: self.epc_mapping.get(row['region'], {}).get(row['epcScore'], None), axis=1)
        df['epcScore'] = df['epcScore'].fillna(0)
        
        # Convert building condition
        condition_rating = {
            'to restore': 0, 'to renovate': 1, 'to be done up': 2,
            'good': 3, 'just renovated': 4, 'as new': 5
        }
        df['buildingCondition'] = (df['buildingCondition'].astype(str).str.strip().str.lower()
                                .map(condition_rating).fillna(-1).astype(int))
        
        # Convert flood zone type
        df['floodZoneType'] = (df['floodZoneType'] != 'NON_FLOOD_ZONE').astype(int)
        
        print(f"Final data shape: {df.shape}")
        return df

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_columns = ['province', 'heatingType', 'kitchenType', 'subtype', 'terraceOrientation']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # One-hot encode categorical columns
        for col in self.categorical_columns:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col, dtype=int)
        
        return df

class CoordinateGetter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        df_giraffe = pd.read_csv('data/Giraffe.csv')
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

class KDEKNNFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, k=20):
        self.k = k
        self.scaler = StandardScaler()
        self.knn = NearestNeighbors(n_neighbors=k)
        self.train_prices = None
        
    def fit(self, X, y=None):
        if 'latitude' not in X.columns or 'longitude' not in X.columns:
            print("Warning: Missing latitude/longitude columns")
            return self
            
        coords_scaled = self.scaler.fit_transform(X[['latitude', 'longitude']])
        self.knn.fit(coords_scaled)
        
        # Store training prices
        self.train_prices = X['pricePerM2'].values
        
        return self
        
    def transform(self, X):
        df = X.copy()
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            print("Warning: Missing latitude/longitude columns")
            df['kde_price_per_m2_knn'] = np.nan
            return df
            
        coords_scaled = self.scaler.transform(df[['latitude', 'longitude']])
        distances, indices = self.knn.kneighbors(coords_scaled)
        
        kde_scores = []
        
        print(f"Processing {len(df)} rows for KDE calculation")
        invalid_kde_count = 0
        
        for i in range(len(df)):
            neighbor_idxs = indices[i]
            # Use stored training prices for neighbors
            neighbor_prices = self.train_prices[neighbor_idxs]
            neighbor_prices = neighbor_prices[~np.isnan(neighbor_prices)]
            
            if len(neighbor_prices) < 2:
                kde_scores.append(np.nan)
                invalid_kde_count += 1
                continue
                
            try:
                kde = gaussian_kde(neighbor_prices)
                value_to_evaluate = df['pricePerM2'].iloc[i] if 'pricePerM2' in df.columns else neighbor_prices.mean()
                kde_score = kde(value_to_evaluate)[0]
                
                if np.isfinite(kde_score):
                    kde_scores.append(kde_score)
                else:
                    kde_scores.append(np.nan)
                    invalid_kde_count += 1
            except Exception as e:
                print(f"Error in KDE calculation for row {i}: {str(e)}")
                kde_scores.append(np.nan)
                invalid_kde_count += 1
        
        print(f"KDE calculation complete. {invalid_kde_count} invalid calculations out of {len(df)} rows")
        
        df['kde_price_per_m2_knn'] = kde_scores
        
        # Fill NaN values with median by region
        median_by_region = df.groupby('region')['kde_price_per_m2_knn'].median()
        df['kde_price_per_m2_knn'] = df.apply(
            lambda row: median_by_region[row['region']] if pd.isna(row['kde_price_per_m2_knn']) else row['kde_price_per_m2_knn'],
            axis=1
        )
        
        return df.drop(columns=['latitude', 'longitude'], errors='ignore')

class ColumnCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop = [
            'id', 'postCode', 'buildingConstructionYear', 'type', 'locality', 'region',
            'latitude', 'longitude'
        ]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # Drop columns that are no longer needed
        columns_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        df = df.drop(columns=columns_to_drop)
        
        # Ensure all remaining columns are numeric
        non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(non_numeric_cols) > 0:
            print(f"Warning: Found non-numeric columns before scaling: {non_numeric_cols}")
            # Convert any remaining categorical columns to numeric
            for col in non_numeric_cols:
                if col != 'price':  # Don't encode the target variable
                    df[col] = pd.Categorical(df[col]).codes
        
        # Reorganize columns to put price at the end
        cols = df.columns.tolist()
        if 'price' in cols:
            cols.remove('price')
            cols.append('price')
            df = df[cols]
            
        return df 