import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Function to load data and labels from a directory
def load_data_from_folder(folder_path):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".particles"):
            if 'world' not in filename:
                label = filename.split("_")[0]  # Assuming label is before the first underscore
                
                file_path = os.path.join(folder_path, filename)
                particle_data = np.loadtxt(file_path)
                
                # Center the particle cloud at (0, 0, 0)
                particle_data -= np.mean(particle_data, axis=0, keepdims=True)
                data.append(particle_data)
                labels.append(label)
    return np.array(data), np.array(labels)

# Function to scale the dataset to [0, 1]
def scale_dataset(data):
    # Calculate scale for each sample (maximum norm of points in the sample)
    scales = np.max(np.linalg.norm(data, axis=2, keepdims=True), axis=1, keepdims=True)
    scaled_data = data / scales  # Scale each sample individually
    return scaled_data

# Load train, test, and val data
#Vanilla Mesh2SSM
train_folder = "lumbar/results/train_correspondences_pred"
test_folder = "lumbar/results/test_correspondences_pred/"
val_folder = "lumbar/results/val_correspondences_pred/"


X_train, y_train = load_data_from_folder(train_folder)
X_test, y_test = load_data_from_folder(test_folder)
X_val, y_val = load_data_from_folder(val_folder)


# Scale the datasets
X_train = scale_dataset(X_train)
X_test = scale_dataset(X_test)
X_val = scale_dataset(X_val)

# Flatten the particle data if necessary (e.g., for classifiers that require 2D input)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)

# Encode labels as integers
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_val = label_encoder.transform(y_val)

X = np.vstack([X_train, X_val, X_test])

labels = np.vstack([y_train.reshape((-1,1)), y_val.reshape((-1,1)), y_test.reshape((-1,1))])
labels = labels[:,0]
# Flatten the particle data for classifier input
data = X.reshape(X.shape[0], -1)
# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
f1_scores = []
class_metrics = []

for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Train the classifier
    classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Collect overall metrics
    accuracies.append(accuracy)
    f1_scores.append(f1)

    # Collect class-wise metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    class_metrics.append(report)

# Report results
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print("CLASSIFICATION ACCURACY: MEAN VALUES OF")
print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")

# Aggregate class-wise metrics
class_wise_metrics = {}
for cls in range(len(np.unique(labels))):  # Assuming classes are 0, 1, ..., n-1
    precision = np.mean([fold[str(cls)]['precision'] for fold in class_metrics])
    recall = np.mean([fold[str(cls)]['recall'] for fold in class_metrics])
    f1 = np.mean([fold[str(cls)]['f1-score'] for fold in class_metrics])
    class_wise_metrics[cls] = {
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
    }

# Print class-wise metrics
print("\nClass-wise Metrics (Averaged over 5 folds):")
for cls, metrics in class_wise_metrics.items():
    print(f"Class {cls}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}")