import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .pipeline_components import (
    DataCleaner, FeatureEngineer, CategoricalEncoder, 
    KDEKNNFeatureCreator, CoordinateGetter, ColumnCleaner
)

def create_preprocessing_pipeline():
    """
    Creates a preprocessing pipeline for real estate data.
    """
    return Pipeline([
        ('data_cleaner', DataCleaner()),
        ('feature_engineer', FeatureEngineer()),
        ('coordinate_getter', CoordinateGetter()),
        ('categorical_encoder', CategoricalEncoder()),
        ('kde_knn_creator', KDEKNNFeatureCreator()),
        ('column_cleaner', ColumnCleaner())
        #('standard_scaler', StandardScaler())
    ])

def preprocess_data(df_train, df_test=None):
    """
    Preprocesses the training and test data using the pipeline.
    
    Args:
        df_train (pd.DataFrame): Training data
        df_test (pd.DataFrame, optional): Test data
        
    Returns:
        tuple: (X_train, X_test) where X_test is None if df_test is None
    """
    # Drop rows with NaN values in critical columns
    df_train = df_train.dropna(subset=['price', 'habitableSurface'])
    if df_test is not None:
        df_test = df_test.dropna(subset=['price', 'habitableSurface'])
    
    pipeline = create_preprocessing_pipeline()
    
    # Fit and transform training data
    X_train = pipeline.fit_transform(df_train)
    
    if df_test is not None:
        # Only transform test data
        X_test = pipeline.transform(df_test)
        return X_train, X_test
    
    return X_train, None

def main():
    # Load data
    df = pd.read_csv("./data/Kangaroo.csv")
    df = df.drop_duplicates(subset=["id"], keep="first")
    df = df[df['price'] < 1500000]
    df = df.dropna(subset=['price'])
    
    # Filter EPC scores
    epc_order = ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    df = df[df['epcScore'].isin(epc_order)]
    df['epcScore'] = df['epcScore'].fillna(df['epcScore'].mode()[0])
    
    # Convert price to float
    df['price'] = df['price'].astype(float)
    
    # Split data
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Preprocess data
    X_train, X_test = preprocess_data(df_train, df_test)
    
    # Save processed data
    pd.DataFrame(X_train).to_csv("./data/train_processed.csv", index=False)
    pd.DataFrame(X_test).to_csv("./data/test_processed.csv", index=False)

if __name__ == "__main__":
    main() 