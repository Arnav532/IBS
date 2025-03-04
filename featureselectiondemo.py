import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
# Set the same parameters as in your main code
param = {
    "Max_A_Size": 10,
    "Max_B_Size": 10,
    "Dynamic_Size": False,
    'Metod': 'tSNE',
    "ValidRatio": 0.1,
    "seed": 180,
    "dir": "D:/IDS/Code/Datasets/NSLKDD/",  # Directory for input files
    "res": "D:/IDS/Code/Results/",  # Directory for output results
    "Mode": "CNN2",  # Model type
    "LoadFromJson": False,  # Force data processing for analysis
    "mutual_info": True,  # Use Mutual Information
    "hyper_opt_evals": 20,
    "epoch": 150,
    "No_0_MI": False,
    "autoencoder": False,
    "cut": None,
    "enhanced_dataset": None
}

# ==================== LOAD ORIGINAL DATA ====================
print("Loading original data to get feature names...")

# Adjust this path to your original dataset
train_data_path = param["dir"] + "NSL_KDD_Train.csv"  # Replace with your actual file name
try:
    original_data = pd.read_csv(train_data_path)
    print(f"Successfully loaded data with {original_data.shape[1]} features")
    
    # Extract feature names (assuming the last column is the target)
    y_column = "classification"  # Change this to match your target column name
    
    # If y_column doesn't exist, try to find it
    if y_column not in original_data.columns:
        potential_target_columns = ["classification", "class", "label", "target", "attack_type"]
        for col in potential_target_columns:
            if col in original_data.columns:
                y_column = col
                print(f"Found target column: {y_column}")
                break
        else:
            # If we still don't find the target, assume it's the last column
            y_column = original_data.columns[-1]
            print(f"Assuming last column as target: {y_column}")
    
    feature_names = original_data.drop(y_column, axis=1).columns.tolist()
    print(f"Target column: {y_column}")
    print(f"Number of features: {len(feature_names)}")
except Exception as e:
    print(f"Error loading original data: {str(e)}")
    print("Using generic feature names instead...")
    # Create generic feature names if file loading fails
    feature_names = [f"feature_{i}" for i in range(100)]  # Adjust the range based on expected number of features
    original_data = None

# ==================== RUN FEATURE SELECTION ANALYSIS ====================
def analyze_feature_selection():
    print("\n" + "="*50)
    print("FEATURE SELECTION ANALYSIS")
    print("="*50)
    
    # Load data or create a subset for demonstration
    if original_data is not None:
        print("Using actual dataset for analysis...")
        X_train = original_data.drop(y_column, axis=1)
        y_train = original_data[y_column]
        
        # Detect and encode categorical features
        categorical_cols = []
        for col in X_train.columns:
            if X_train[col].dtype == 'object' or X_train[col].nunique() < 10:
                categorical_cols.append(col)
        
        print(f"Detected {len(categorical_cols)} categorical features")
        
        # Create a copy for encoding
        X_train_encoded = X_train.copy()
        
        # Apply label encoding to categorical columns
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
            label_encoders[col] = le
        
        print(f"Encoded categorical features using Label Encoding")
        
        # Proceed with encoded data
        X_processed = X_train_encoded
        
    else:
        print("Using synthetic data for demonstration...")
        # Create synthetic data if loading fails
        np.random.seed(42)
        X_processed = pd.DataFrame(np.random.random((1000, len(feature_names))), columns=feature_names)
        y_train = pd.Series(np.random.randint(0, 5, 1000))
    
    print(f"Processed data shape: {X_processed.shape}")
    
    # Calculate mutual information between features and target
    print("Calculating mutual information scores...")
    mi_scores = mutual_info_classif(X_processed, y_train)
    mi_df = pd.DataFrame({'Feature': feature_names, 'Mutual_Information': mi_scores})
    mi_df = mi_df.sort_values('Mutual_Information', ascending=False)
    
    # Print top and bottom features by mutual information
    print("\nTop 10 features by mutual information:")
    print(mi_df.head(10))
    
    print("\nBottom 10 features by mutual information:")
    print(mi_df.tail(10))
    
    # Visualize mutual information scores
    plt.figure(figsize=(12, 8))
    top_features = mi_df.head(20)
    sns.barplot(x='Mutual_Information', y='Feature', data=top_features)
    plt.title('Top 20 Features by Mutual Information')
    plt.tight_layout()
    plt.show()
    
    # ==================== SIMULATE CART2PIXEL FEATURE SELECTION ====================
    print("\nSimulating Cart2Pixel feature selection process...")
    
    # Prepare data for t-SNE (use a subset if too large)
    sample_size = min(5000, X_processed.shape[0])
    if len(X_processed) > sample_size:
        indices = np.random.choice(len(X_processed), sample_size, replace=False)
        X_sample = X_processed.iloc[indices]
        y_sample = y_train.iloc[indices]
    else:
        X_sample = X_processed
        y_sample = y_train
    
    # Run dimensionality reduction
    try:
        from sklearn.manifold import TSNE
        print("Applying t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=param["seed"])
        X_2d = tsne.fit_transform(X_sample)
        
        # Visualize t-SNE result
        plt.figure(figsize=(10, 8))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_sample, cmap='viridis', alpha=0.5)
        plt.title('t-SNE Visualization of Features')
        plt.colorbar(label='Class')
        plt.show()
        
        # Simulate coordinate transformation
        print("Transforming to pixel coordinates...")
        A, B = param["Max_A_Size"], param["Max_B_Size"]
        
        # Scale to pixel coordinates (simplified)
        xp = np.round(1 + (A * (X_2d[:, 0] - X_2d[:, 0].min()) / (X_2d[:, 0].max() - X_2d[:, 0].min())))
        yp = np.round(1 + (B * (X_2d[:, 1] - X_2d[:, 1].min()) / (X_2d[:, 1].max() - X_2d[:, 1].min())))
        
        # Visualize pixel mapping
        plt.figure(figsize=(10, 8))
        plt.scatter(xp, yp, c=y_sample, cmap='viridis', alpha=0.5)
        plt.title('Pixel Mapping of Features')
        plt.xlabel('X Pixel Coordinate')
        plt.ylabel('Y Pixel Coordinate')
        plt.grid(True)
        plt.colorbar(label='Class')
        plt.show()
        
        # Find duplicate pixel locations
        print("Finding duplicate feature mappings...")
        duplicates = {}
        for i in range(len(xp)):
            pixel_key = f"{int(xp[i])}-{int(yp[i])}"
            if pixel_key not in duplicates:
                duplicates[pixel_key] = []
            duplicates[pixel_key].append(i)
        
        # Filter to keep only locations with duplicates
        collision_points = {k: v for k, v in duplicates.items() if len(v) > 1}
        
        print(f"Found {len(collision_points)} pixel locations with collisions")
        print(f"Total features in collisions: {sum(len(v) for v in collision_points.values())}")
        
        # Map sample indices back to feature indices for easier interpretation
        # We're operating on a sample of the data, so we need to map the indices
        feature_indices = list(range(len(feature_names)))
        kept_features = set(feature_indices)
        discarded_features = set()
        
        # Simulate mutual information selection at each collision point
        for loc, sample_indices in collision_points.items():
            if len(sample_indices) > 1:  # If there's a collision
                collision_features = [feature_names[i % len(feature_names)] for i in sample_indices]
                collision_mi = [mi_scores[feature_names.index(f)] for f in collision_features]
                
                # Keep the feature with highest MI
                best_idx = sample_indices[np.argmax(collision_mi)]
                best_feature = feature_names[best_idx % len(feature_names)]
                
                # Discard others
                for idx in sample_indices:
                    if idx != best_idx:
                        feature = feature_names[idx % len(feature_names)]
                        if feature in kept_features:
                            kept_features.remove(feature)
                            discarded_features.add(feature)
        
        # Final lists of kept and discarded features
        kept_feature_names = list(kept_features)
        discarded_feature_names = list(discarded_features)
        
        print(f"\nRESULTS:")
        print(f"Total features: {len(feature_names)}")
        print(f"Kept features: {len(kept_feature_names)}")
        print(f"Discarded features: {len(discarded_feature_names)}")
        
        # Display sample of kept and discarded features
        print("\nSample of kept features:")
        for i, feat in enumerate(sorted(kept_feature_names)[:20]):
            print(f"{i+1}. {feat}")
            
        print("\nSample of discarded features:")
        for i, feat in enumerate(sorted(discarded_feature_names)[:20]):
            print(f"{i+1}. {feat}")
        
        # Visualize feature selection
        kept_mi = [mi_scores[feature_names.index(f)] for f in kept_feature_names if feature_names.index(f) < len(mi_scores)]
        discarded_mi = [mi_scores[feature_names.index(f)] for f in discarded_feature_names if feature_names.index(f) < len(mi_scores)]
        
        plt.figure(figsize=(10, 6))
        plt.hist(kept_mi, alpha=0.5, label='Kept Features', bins=20)
        plt.hist(discarded_mi, alpha=0.5, label='Discarded Features', bins=20)
        plt.xlabel('Mutual Information')
        plt.ylabel('Count')
        plt.title('Distribution of Mutual Information for Kept vs Discarded Features')
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"Error during t-SNE or collision analysis: {str(e)}")
        print("Falling back to feature importance only...")
        kept_feature_names = mi_df['Feature'].head(int(len(feature_names) * 0.8)).tolist()
        discarded_feature_names = mi_df['Feature'].tail(int(len(feature_names) * 0.2)).tolist()
    
    return kept_feature_names, discarded_feature_names

# ==================== RUN THE ANALYSIS ====================
try:
    # Call the analysis function
    kept_features, discarded_features = analyze_feature_selection()

    # Print final summary
    print("\n" + "="*50)
    print("FEATURE SELECTION SUMMARY")
    print("="*50)
    print(f"Total original features: {len(feature_names)}")
    print(f"Features retained: {len(kept_features)} ({len(kept_features)/len(feature_names)*100:.1f}%)")
    print(f"Features discarded: {len(discarded_features)} ({len(discarded_features)/len(feature_names)*100:.1f}%)")

    # Save results to file
    results = {
        'original_features': feature_names,
        'kept_features': kept_features,
        'discarded_features': discarded_features
    }

    try:
        with open(param["res"] + 'feature_selection_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"\nFeature selection results saved to {param['res']}feature_selection_results.pkl")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
except Exception as e:
    print(f"Analysis error: {str(e)}")

print("\nTo access actual feature selection results from the training process,")
print("check the 'toDelete' variable returned by Cart2Pixel in the training code.")