# -*- coding: utf-8 -*-
"""
Data processing module for NSL-KDD dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import matplotlib.pyplot as plt

def loadData(fromPath, LabelColumnName, labelCount):
    """
    Load NSL-KDD dataset and show initial statistics
    
    Args:
        fromPath: Path to NSL-KDD CSV file
        LabelColumnName: Name of the label column
        labelCount: Not used but kept for compatibility
    """
    try:
        print(f"\nLoading dataset from: {fromPath}")
        data_ = pd.read_csv(fromPath)
        dataset = data_
        
        # Get and display class distribution
        data = dataset[LabelColumnName].value_counts()
        print("\nDataset Summary:")
        print(f"Total samples: {len(dataset)}")
        print("\nClass distribution:")
        for label, count in data.items():
            print(f"{label}: {count}")
        
        # Get feature list excluding label column
        featureList = dataset.drop([LabelColumnName], axis=1).columns
        print(f"\nNumber of features: {len(featureList)}")
        
        return dataset, featureList
        
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        raise

def datasetSplit(df, LabelColumnName):
    """
    Split dataset into features and labels, handle preprocessing
    
    Args:
        df: Input DataFrame
        LabelColumnName: Name of the label column
    """
    try:
        print("\nStarting data preprocessing...")
        
        # First encode the label column
        labelencoder = LabelEncoder()
        df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
        
        # Get features (all columns except label)
        X = df.drop([LabelColumnName], axis=1)
        
        # Identify categorical and numeric columns
        categorical_columns = ['protocol_type', 'service', 'flag']
        numeric_columns = [col for col in X.columns if col not in categorical_columns]
        
        print("\nPreprocessing features:")
        print(f"Categorical columns: {categorical_columns}")
        print(f"Number of numeric columns: {len(numeric_columns)}")
        
        # Encode categorical columns
        for column in categorical_columns:
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].astype(str))
            unique_values = len(np.unique(X[column]))
            print(f"Encoded {column}: {unique_values} unique values")
        
        # Convert to numpy array
        X = X.values
        X = X.T
        
        # Handle missing/infinite values only for numeric columns
        print("\nHandling missing and infinite values...")
        for i, column in enumerate(X):
            if i not in range(X.shape[0] - len(categorical_columns), X.shape[0]):
                column = column.astype(float)
                median = np.nanmedian(column)
                column[np.isnan(column)] = median
                column[column == np.inf] = 0
                column[column == -np.inf] = 0
        X = X.T
        
        # Scale features
        print("\nScaling features...")
        scaler = preprocessing.MinMaxScaler()
        X = scaler.fit_transform(X)
        
        y = df[[LabelColumnName]]
        
        print("\nProcessing complete:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        return X, y
        
    except Exception as e:
        print(f"\nError in datasetSplit:")
        print(f"Error message: {str(e)}")
        print("\nColumn types before processing:")
        print(df.dtypes)
        raise

def train_test_dataset(X, y):
    """
    Split data into train and test sets
    
    Args:
        X: Feature matrix
        y: Labels
    """
    try:
        print("\nSplitting into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=0.7,
            test_size=0.3,
            random_state=42,
            stratify=y
        )
        
        print("\nSplit sizes:")
        print(f"Training: {X_train.shape[0]} samples")
        print(f"Testing: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"\nError in train_test_split: {str(e)}")
        raise

def plot_class_distribution(y, title="Class Distribution"):
    """
    Plot the distribution of classes in the dataset
    """
    plt.figure(figsize=(12, 6))
    y.value_counts().plot(kind='bar')
    plt.title(title)
    plt.xlabel("Attack Type")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Example usage and testing
if __name__ == "__main__":
    try:
        # Test the data processing pipeline
        fromPath = "D:/IDS/NSL-KDD-Network-Intrusion-Detection/NSL_KDD_Train.csv"
        LabelColumnName = 'classification'
        
        # Load data
        print("Testing data loading...")
        dataset, featureList = loadData(fromPath, LabelColumnName, 4)
        
        # Plot initial distribution
        plot_class_distribution(dataset[LabelColumnName], "Initial Class Distribution")
        
        # Split and preprocess
        print("\nTesting data preprocessing...")
        X, y = datasetSplit(dataset, LabelColumnName)
        
        # Create train/test split
        print("\nTesting train/test split...")
        X_train, X_test, y_train, y_test = train_test_dataset(X, y)
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise