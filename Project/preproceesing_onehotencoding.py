import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Function to map latency to a categorical variable
def map_latency(latency):
    if latency == '<10ms':
        return 'latency_10ms'
    elif latency == '<50ms':
        return 'latency_50ms'
    elif latency == '<300ms':
        return 'latency_300ms'
    else:
        return 'latency_unknown'  # For any other case not covered

# Load the dataset from a specified sheet
def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

# Preprocess the dataset
def preprocess_data(data):
    # Drop the 'Packet Loss Rate (Reliability)' column
    data.drop('Packet Loss Rate (Reliability)', axis=1, inplace=True)
    
    # Map latency to categorical
    data['Packet Delay Budget (Latency)'] = data['Packet Delay Budget (Latency)'].apply(map_latency)
    
    # Drop the original 'Packet Delay Budget (Latency)' column as it will be one-hot encoded
    data.drop('Packet Delay Budget (Latency)', axis=1, inplace=True)

    # Fill missing values in categorical columns with mode
    categorical_cols = ['Use CaseType (Input 1)', 'Day (Input4)', 'Slice Type (Output)', 'LTE/5G UE Category (Input 2)', 'Technology Supported (Input 3)']
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Numeric columns to normalize
    numeric_cols = ['Time (Input 5)', 'QCI (Input 6)']  # Assuming these are the only numeric columns left
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=categorical_cols)

    # Normalize numerical features
    scaler = MinMaxScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data

# Main function to run the steps
def main():
    file_path = r"C:\Users\Cherry\OneDrive\Desktop\Machine Learning\project\5G_Dataset_Network_Slicing_CRAWDAD_Shared (1).xlsx"
    sheet_name = 'Model_Inputs_Outputs'
    data = load_data(file_path, sheet_name)
    cleaned_data = preprocess_data(data)

    # Save the preprocessed data to a new CSV file
    output_path = r"C:\Users\Cherry\OneDrive\Desktop\Machine Learning\project\processed_data_without_packet_loss_delay.csv"
    cleaned_data.to_csv(output_path, index=False)
    print("Data preprocessed and saved to:", output_path)

# Call the main function
if __name__ == '__main__':
    main()
