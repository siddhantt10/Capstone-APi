from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target  # Features and target
classes = data.target_names  # 'malignant', 'benign'

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Apply PCA for dimensionality reduction (optional)
# pca = PCA(n_components=10)  # Reduce to 10 components (you can adjust this)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

# SVM Model with Grid Search for Hyperparameter Tuning
parameters = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'class_weight': ['balanced', None]  # Adjust for class imbalance
}

grid_search = GridSearchCV(SVC(), parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_model = grid_search.best_estimator_

# Model Testing
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Best Model: {grid_search.best_params_}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

def predBreastCancer(sample):
    sample = scaler.transform([sample])
    # sample = pca.transform(sample)  # Apply PCA transformation
    prediction = best_model.predict(sample)
    return classes[prediction[0]]

# Tests
malignant = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
benign = [
    12.77, 18.02, 81.35, 492.1, 0.08432, 0.09769, 0.1036, 0.05302, 0.1734, 0.06148, 
    0.3729, 0.8755, 2.567, 27.91, 0.007617, 0.01464, 0.0234, 0.01241, 0.01559, 0.003732, 
    14.03, 24.9, 89.59, 582.7, 0.1106, 0.2135, 0.2706, 0.09926, 0.294, 0.08452
]
result = predBreastCancer(malignant)
print(f"The prediction for the sample is: {result}")
