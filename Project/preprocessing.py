import pandas as pd
import numpy as np

df = pd.read_csv('Dataset/5G_Dataset_Network_Slicing_CRAWDAD_Dataset.csv', dtype=str)

df = df.drop(df.columns[0], axis=1)     # Deleting the empty column
df = df.astype(str)


# Transforming the categorical labels to numerical categories
factor = pd.factorize(df['Slice Type (Output)'])
y = np.array(factor[0])
labels = factor[1]

df = df.drop(df.columns[8], axis=1)

X = []
X_features = []

# Transforming the categorical features to numerical categories
for col in df:
    fact = pd.factorize(df[col])
    X.append(fact[0])
    X_features.append(fact[1])

X = np.array(X).T

np.savetxt('X_data.csv', X, delimiter=',', fmt='%d')
np.savetxt('y_data.csv', y, delimiter=',', fmt='%d')
np.savetxt('labels.csv', labels, delimiter=',', fmt='%s')

# Splitting the dataset to features and labels
print("Data preprocessing complete.")
print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])