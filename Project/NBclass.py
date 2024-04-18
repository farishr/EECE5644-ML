import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

data_path = r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\Project\Processed.csv'
df = pd.read_csv(data_path)

# list of columns to drop
columns_to_drop = [col for col in df.columns if 'Packet Delay Budget (Latency)' in col or 'Packet Loss Rate (Reliability)' in col]

# Rest of the features as X and the target variable 'Slice Type (Output)' as y
target_columns = ['Slice Type (Output)_URLLC', 'Slice Type (Output)_eMBB', 'Slice Type (Output)_mMTC']
X = df.drop(columns_to_drop + target_columns, axis=1)
y = df[target_columns].idxmax(axis=1)

# binarizing the output
class_labels = target_columns
y_bin = label_binarize(y, classes=class_labels)
n_classes = y_bin.shape[1]

# splitting data into training and testing sets (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)

classifier = OneVsRestClassifier(GaussianNB())

# training the model
classifier.fit(X_train, y_train)

# predicting on the test set
y_pred = classifier.predict(X_test)
y_score = classifier.predict_proba(X_test)

accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(f'Accuracy: {accuracy:.2f}')

roc_auc_ovr = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
print(f'ROC AUC Score (One-vs-Rest): {roc_auc_ovr:.2f}')

# computing ROC curve and ROC AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'green', 'red'])
for i, color in zip(range(n_classes), colors):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve for class {class_labels[i].split("_")[-1]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# building confusion matrix heatmap
conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
class_names = [label.split('_')[-1] for label in class_labels]  # Extract class names for labels
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()