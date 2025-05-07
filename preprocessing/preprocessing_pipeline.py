import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import pgeocode

class EPCTransformer(BaseEstimator, TransformerMixin):
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
        X = X.copy()
        X['region'] = X['postCode'].apply(self._get_region)
        X['epcScore'] = X.apply(lambda row: self.epc_mapping.get(row['region'], {}).get(row['epcScore']), axis=1)
        return X
    
    def _get_region(self, zip_code):
        if 1000 <= zip_code <= 1299:
            return "Bruxelles"
        elif 1300 <= zip_code <= 1499 or 4000 <= zip_code <= 7999:
            return "Wallonia"
        else:
            return "Flanders"

class BuildingConditionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.condition_rating = {
            'to restore': 0,
            'to renovate': 1,
            'to be done up': 2,
            'good': 3,
            'just renovated': 4,
            'as new': 5
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['buildingCondition'] = (X['buildingCondition']
                                .astype(str)
                                .str.strip()
                                .str.lower()
                                .map(self.condition_rating)
                                .fillna(-1)
                                .astype(int))
        return X

class PricePerM2Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['pricePerM2'] = X['price'] / X['habitableSurface']
        return X

class CoordinateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, use_giraffe=True):
        self.use_giraffe = use_giraffe
        self.nomi = None
        self.giraffe_data = None
    
    def fit(self, X, y=None):
        if not self.use_giraffe:
            self.nomi = pgeocode.Nominatim('be')
        else:
            self.giraffe_data = pd.read_csv('data/Giraffe.csv')
            self.giraffe_data = self.giraffe_data[['propertyId', 'latitude', 'longitude']]
            self.giraffe_data['id'] = self.giraffe_data['propertyId']
            self.giraffe_data = self.giraffe_data.drop(columns='propertyId')
        return self
    
    def transform(self, X):
        X = X.copy()
        if not self.use_giraffe:
            X['postCode'] = X['postCode'].astype(str)
            unique_postcodes = X["postCode"].astype(str).unique()
            
            geo_df = self.nomi.query_postal_code(list(unique_postcodes))
            geo_df = geo_df[['postal_code', 'latitude', 'longitude']]
            geo_df = geo_df.rename(columns={'postal_code': 'postCode'})
            geo_df['postCode'] = geo_df['postCode'].astype(str)
            
            X = X.merge(geo_df, on='postCode', how='left')
        else:
            X = X.merge(self.giraffe_data, on='id', how='left')
        
        X = X.dropna(subset=['latitude', 'longitude'])
        return X

class KDEScorer(BaseEstimator, TransformerMixin):
    def __init__(self, k=20):
        self.k = k
        self.scaler = None
        self.knn = None
        self.train_data = None
        self.price_per_m2 = None
    
    def fit(self, X, y=None):
        self.train_data = X.copy()
        self.price_per_m2 = X['pricePerM2'].values
        
        coords = X[['latitude', 'longitude']].values
        self.scaler = StandardScaler()
        coords_scaled = self.scaler.fit_transform(coords)
        
        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.knn.fit(coords_scaled)
        return self
    
    def transform(self, X):
        X = X.copy()
        coords = X[['latitude', 'longitude']].values
        coords_scaled = self.scaler.transform(coords)
        distances, indices = self.knn.kneighbors(coords_scaled)
        
        kde_scores = []
        for i in range(len(X)):
            neighbor_idxs = indices[i]
            neighbor_prices = self.price_per_m2[neighbor_idxs]
            neighbor_prices = neighbor_prices[~np.isnan(neighbor_prices)]
            
            if len(neighbor_prices) < 2:
                kde_scores.append(np.nan)
            else:
                try:
                    kde = gaussian_kde(neighbor_prices)
                    value_to_evaluate = X['pricePerM2'].iloc[i]
                    kde_scores.append(kde(value_to_evaluate)[0])
                except (ValueError, np.linalg.LinAlgError):
                    # If KDE fails, use the mean of neighbors
                    kde_scores.append(neighbor_prices.mean())
        
        X['kde_price_per_m2_knn'] = kde_scores
        # Fill any remaining NaN values with the median
        X['kde_price_per_m2_knn'] = X['kde_price_per_m2_knn'].fillna(X['kde_price_per_m2_knn'].median())
        return X

class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, feature_names):
        self.transformer = transformer
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        self.transformer.fit(X)
        return self
    
    def transform(self, X):
        transformed = self.transformer.transform(X)
        return pd.DataFrame(transformed, columns=self.feature_names)

def create_preprocessing_pipeline():
    # Define feature groups
    numeric_features = [
        'habitableSurface', 'buildingConstructionYear', 'landSurface',
        'livingRoomSurface', 'gardenSurface', 'terraceSurface'
    ]
    
    categorical_features = [
        'province', 'heatingType', 'kitchenType', 'subtype',
        'terraceOrientation'
    ]
    
    binary_features = [
        'hasBasement', 'hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels',
        'hasAirConditioning', 'hasArmoredDoor', 'hasVisiophone', 'hasOffice',
        'hasSwimmingPool', 'hasFireplace', 'hasAttic', 'hasLivingRoom',
        'hasGarden', 'hasTerrace'
    ]
    
    location_features = ['latitude', 'longitude']
    
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create the main pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('binary', 'passthrough', binary_features),
            ('location', 'passthrough', location_features),
            ('price', 'passthrough', ['price'])  # Explicitly include price column
        ]
    )
    
    # Create the full pipeline
    pipeline = Pipeline([
        ('coordinates', CoordinateTransformer(use_giraffe=True)),  # Add coordinates first
        ('drop_coord_nulls', DropNullsTransformer(['latitude', 'longitude'])),  # Drop rows with missing coordinates
        ('epc', EPCTransformer()),
        ('building_condition', BuildingConditionTransformer()),
        ('price_per_m2', PricePerM2Transformer()),  # Calculate price per m2
        ('kde_scorer', KDEScorer()),  # Do KDE scoring while we still have DataFrame
        ('preprocessor', preprocessor)  # Do the final preprocessing
    ])
    
    return pipeline

def preprocess_data():
    # Load and initial cleaning
    df = pd.read_csv("./data/Kangaroo.csv")
    df = df.drop_duplicates(subset=["id"], keep="first")
    
    # Drop rows with NaN values in critical columns
    critical_columns = ['price', 'habitableSurface']  # Only check these initially
    df = df.dropna(subset=critical_columns)
    
    # Filter out extreme prices
    df = df[df['price'] < 1000000]
    
    # Filter valid EPC scores
    epc_order = ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    df = df[df['epcScore'].isin(epc_order)]
    df['epcScore'] = df['epcScore'].fillna(df['epcScore'].mode()[0])
    
    # Convert price to float
    df['price'] = df['price'].astype(float)
    
    return df

def train_test_split_data(df):
    return train_test_split(df, test_size=0.2, random_state=42)

def getData():
    # Load and preprocess data
    df = preprocess_data()
    df_train, df_test = train_test_split_data(df)
    
    # Create and fit pipeline
    pipeline = create_preprocessing_pipeline()
    pipeline.fit(df_train)
    
    # Transform data
    df_train_processed = pipeline.transform(df_train)
    df_test_processed = pipeline.transform(df_test)
    
    # Get feature names from the preprocessor
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # Convert numpy arrays back to DataFrames with proper column names
    df_train_processed = pd.DataFrame(df_train_processed, columns=feature_names)
    df_test_processed = pd.DataFrame(df_test_processed, columns=feature_names)
    
    # Prepare data for modeling
    X_train, y_train, X_test, y_test = prepare_modeling_data(df_train_processed, df_test_processed)
    
    # Save processed data
    df_train_processed.to_csv("./data/train.csv", index=False)
    df_test_processed.to_csv("./data/test.csv", index=False)
    
    return X_train, y_train, X_test, y_test

class DropNullsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        return X.dropna(subset=self.columns)

def prepare_modeling_data(df_train_processed, df_test_processed):
    # Prepare training data
    X_train = df_train_processed.drop(columns=['price'])
    y_train = df_train_processed['price']
    
    # Prepare test data
    X_test = df_test_processed.drop(columns=['price'])
    y_test = df_test_processed['price']
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    getData() 