import pandas as pd

def filter_non_benign(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if 'Label' column exists
    if 'classification' not in df.columns:
        print("Error: 'Label' column not found in the CSV file.")
        return
    
    
    # Filter rows where Label is not 'BENIGN'
    benign_data=df[df['classification']=='normal']
    countbenign=len(benign_data)
    print("Number of benign entries:",countbenign)
    non_benign_data = df[df['classification'] != 'normal']
    countnonbenign=len(non_benign_data)
    print("Number of non-benign entries:",countnonbenign)
    # Print the filtered data
    seen=[]
    for i in non_benign_data['classification']:
        if i not in seen:
            seen.append(i)
    else:
        print("No non-BENIGN entries found.")
    print(seen)
    for i in seen:
        print(i,":",len(non_benign_data[non_benign_data['classification']==i]))
    print("\n",len(seen))
# Example usage
csv_file="D:/IDS/NSL-KDD-Network-Intrusion-Detection/NSL_KDD_Train.csv" 
filter_non_benign(csv_file)
