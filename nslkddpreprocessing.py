import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

def load_nsl_kdd_data(filepath, multiclass=True):
    """
    Load and preprocess NSL-KDD dataset
    Args:
        filepath: Path to the NSL-KDD dataset CSV
        multiclass: If True, keep all 23 attack types. If False, binary classification
    """
    print("Loading NSL-KDD dataset...")
    
    # Define column names
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
              'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
              'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
              'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
              'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
              'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
              'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
              'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
              'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
              'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
              'dst_host_srv_rerror_rate','classification.']
    
    # Read the dataset
    data = pd.read_csv(filepath, names=columns)
    
    # List of all attack types for verification
    attack_types = ['normal', 'neptune', 'warezclient', 'ipsweep', 'portsweep', 
                    'teardrop', 'nmap', 'satan', 'smurf', 'pod', 'back', 
                    'guess_passwd', 'ftp_write', 'multihop', 'rootkit', 
                    'buffer_overflow', 'imap', 'warezmaster', 'phf', 'land', 
                    'loadmodule', 'spy', 'perl']
    
    # Verify and print found attack types
    found_attacks = data['classification.'].unique()
    print("\nFound attack types in dataset:", sorted(found_attacks))
    
    # Handle categorical features
    categorical_columns = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    
    # Handle labels
    label_encoder = LabelEncoder()
    data['classification'] = label_encoder.fit_transform(data['classification'])
    
    if not multiclass:
        # Convert to binary classification (normal vs attack)
        data['classification'] = data['classification'].apply(lambda x: 0 if x == 0 else 1)
    
    # Split features and labels
    X = data.drop('classification', axis=1)
    y = data['classification']
    
    # Handle missing values if any
    X = X.fillna(0)
    
    # Normalize numeric features
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
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
    
    # Print detailed dataset information
    print("\nDataset Summary:")
    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {num_classes}")
    
    print("\nClass distribution:")
    for i, label in enumerate(label_encoder.classes_):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"{label}: {train_count} train, {test_count} test")
    
    # Save mapping information
    class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    print("\nClass mapping:")
    for idx, label in class_mapping.items():
        print(f"Class {idx}: {label}")
    
    return dataset, columns[:-1], label_encoder, class_mapping

def print_feature_info(feature_list):
    """
    Print information about the features in the NSL-KDD dataset
    """
    features_info = {
        'basic_features': [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent'
        ],
        'content_features': [
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login'
        ],
        'time_based_features': [
            'Count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate'
        ],
        'host_based_features': [
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
    }
    
    print("\nFeature Groups in NSL-KDD dataset:")
    for group, features in features_info.items():
        print(f"\n{group.replace('_', ' ').title()}:")
        for feature in features:
            if feature in feature_list:
                print(f"- {feature}")