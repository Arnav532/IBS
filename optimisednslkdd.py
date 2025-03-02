import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical

def preprocess_numeric_features(df, numeric_columns):
    """Optimize numeric feature preprocessing for NSL-KDD"""
    scaler = MinMaxScaler()  # Changed from StandardScaler for better t-SNE performance
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df, scaler

def preprocess_categorical_features(df, categorical_columns):
    """Enhanced categorical feature preprocessing"""
    encoders = {}
    for column in categorical_columns:
        # One-hot encoding for service and flag, label encoding for protocol
        if column in ['service', 'flag']:
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df.drop(column, axis=1, inplace=True)
        else:
            encoders[column] = LabelEncoder()
            df[column] = encoders[column].fit_transform(df[column])
    return df, encoders

def load_nsl_kdd_data(filepath, multiclass=True):
    """Enhanced NSL-KDD dataset loading with optimized preprocessing"""
    print("Loading NSL-KDD dataset...")
    
    # Read the dataset with column names
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
              'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
              'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
              'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
              'num_access_files', 'num_outbound_cmds', 'is_host_login', 
              'is_guest_login', 'count', 'srv_count', 'serror_rate', 
              'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
              'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
              'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
              'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
              'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
              'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
              'dst_host_srv_rerror_rate', 'classification']

    # Define column types
    categorical_columns = ['protocol_type', 'service', 'flag']
    binary_columns = ['land', 'logged_in', 'root_shell', 'su_attempted',
                     'is_host_login', 'is_guest_login']
    numeric_columns = [col for col in columns if col not in categorical_columns + binary_columns + ['classification']]
    
    # Read the dataset
    data = pd.read_csv(filepath)
    
    print("\nInitial data shape:", data.shape)
    print("Attack types found:", sorted(data['classification'].unique()))
    
    # Handle binary features
    for col in binary_columns:
        data[col] = data[col].astype(int)
    
    # Preprocess categorical features
    data, cat_encoders = preprocess_categorical_features(data, categorical_columns)
    
    # Preprocess numeric features
    numeric_data = data[numeric_columns]
    data[numeric_columns], num_scaler = preprocess_numeric_features(numeric_data, numeric_columns)
    
    # Handle labels
    label_encoder = LabelEncoder()
    data['classification'] = label_encoder.fit_transform(data['classification'])
    
    if not multiclass:
        # Convert to binary classification (normal vs attack)
        data['classification'] = data['classification'].apply(lambda x: 0 if x == 0 else 1)
    
    # Split features and labels
    X = data.drop('classification', axis=1)
    y = data['classification']
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Convert to numpy arrays
    dataset = {
        "Xtrain": np.array(X_train),
        "Xtest": np.array(X_test),
        "Classification": np.array(y_train),  # Not categorical yet
        "Ytest": np.array(y_test)  # Not categorical yet
    }
    
    preprocessing_info = {
        'categorical_encoders': cat_encoders,
        'numeric_scaler': num_scaler,
        'label_encoder': label_encoder,
        'feature_columns': {
            'numeric': numeric_columns,
            'categorical': categorical_columns,
            'binary': binary_columns
        }
    }
    
    # Print dataset statistics
    print("\nDataset shapes:")
    for key, value in dataset.items():
        print(f"{key}: {value.shape}")
    
    print(f"\nNumber of classes: {len(np.unique(y))}")
    print(f"Features: {X_train.shape[1]}")
    print("\nClass distribution:")
    classes = label_encoder.classes_
    for i, attack_type in enumerate(classes):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"{attack_type}: {train_count} train, {test_count} test")
    
    return dataset, preprocessing_info