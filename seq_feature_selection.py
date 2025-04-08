import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Przygotowanie danych
def prepare_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Obliczanie F-score
def calculate_f_score(X, y):
    classes = np.unique(y)
    f_scores = []
    
    for feature in range(X.shape[1]):
        numerator = 0
        denominator = 0
        
        for c in classes:
            class_samples = X[y == c, feature]
            class_mean = np.mean(class_samples)
            class_var = np.var(class_samples)
            total_mean = np.mean(X[:, feature])
            
            numerator += (class_mean - total_mean) ** 2
            denominator += class_var
            
        f_scores.append(numerator / denominator)
        
    return np.array(f_scores)

# Selekcja cech
def select_features(X, y, n_features=10):
    f_scores = calculate_f_score(X, y)
    selected_features = np.argsort(f_scores)[::-1][:n_features]
    return selected_features