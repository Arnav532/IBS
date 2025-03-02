import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical

def load_cicids_data(filepath, multiclass=True):
    """
    Load and preprocess CICIDS2017 dataset
    Args:
        filepath: Path to the CICIDS dataset CSV
        multiclass: If True, keep all attack types. If False, binary classification
    Returns:
        dataset: Dictionary containing Xtrain, Xtest, Classification, Ytest
        feature_list: List of feature names
        label_encoder: Fitted LabelEncoder object
    """
    print("Loading CICIDS dataset...")
    data = pd.read_csv(filepath)
    
    # Handle label column
    label_column = ' Label'  # Adjust if your column name is different
    
    if not multiclass:
        # Binary classification: Normal vs Attack
        data[label_column] = data[label_column].apply(
            lambda x: 'Normal' if x == 'BENIGN' else 'Attack'
        )
    
    # Get feature list excluding label
    feature_list = data.drop([label_column], axis=1).columns
    
    # Encode labels
    label_encoder = LabelEncoder()
    data[label_column] = label_encoder.fit_transform(data[label_column])
    
    # Split features and labels
    X = data.drop([label_column], axis=1)
    y = data[label_column]
    
    # Handle missing/infinite values in features
    X = X.replace([np.inf, -np.inf], np.nan)
    for column in X.columns:
        median = X[column].median()
        X[column] = X[column].fillna(median)
    
    # Normalize features
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42,
        stratify=y
    )
    
    # Convert labels to categorical
    num_classes = len(np.unique(y))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Create dataset dictionary compatible with existing code
    dataset = {
        "Xtrain": X_train,
        "Xtest": X_test,
        "Classification": y_train_cat,
        "Ytest": y_test_cat
    }
    
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Number of classes: {num_classes}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Class distribution: ")
    for i, label in enumerate(label_encoder.classes_):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"{label}: {train_count} train, {test_count} test")
    
    return dataset, feature_list, label_encoder

def datasetSplit(df, LabelColumnName):
    """
    Alternative method for dataset splitting without train/test split
    Args:
        df: Input dataframe
        LabelColumnName: Name of the label column
    Returns:
        X: Processed features
        y: Processed labels
    """
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    
    X = df.drop([LabelColumnName], axis=1)
    X = np.array(X)
    X = X.T
    
    # Handle missing/infinite values
    for column in X:
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    
    X = X.T
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    
    y = df[[LabelColumnName]]
    return X, y

def train_test_dataset(df, LabelColumnName):
    """
    Alternative method for train/test splitting
    Args:
        df: Input dataframe
        LabelColumnName: Name of the label column
    Returns:
        X_train, X_test, y_train, y_test: Split and processed datasets
    """
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    
    X = df.drop([LabelColumnName], axis=1)
    y = df[[LabelColumnName]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=0.7,
        test_size=0.3,
        random_state=0,
        stratify=y
    )
    
    # Process training data
    X_train = np.array(X_train)
    X_train = X_train.T
    for column in X_train:
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    X_train = X_train.T
    
    # Process test data
    X_test = np.array(X_test)
    X_test = X_test.T
    for column in X_test:
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    X_test = X_test.T
    
    # Process labels
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return X_train, X_test, y_train, y_test