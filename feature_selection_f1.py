import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def evaluate_feature_set(X, y, features):
    """
    Evaluate a feature set using cross-validation with F1 score
    
    Parameters:
    X : DataFrame or array
        The full feature matrix
    y : array
        Target values
    features : list
        List of selected feature indices or names
    
    Returns:
    float : Average cross-validation F1 score
    """
    # Select only the specified features
    X_selected = X[:, features] if isinstance(X, np.ndarray) else X[features]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Initialize classifier
    classifier = SVC(kernel='linear')
    
    # Perform cross-validation with F1 score
    scores = cross_val_score(
        classifier,
        X_scaled,
        y,
        cv=5,
        scoring='f1'
    )
    
    return np.mean(scores)

# Generate example data
X, y = make_classification(
    n_samples=100,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize classifier
clf = SVC(kernel='linear', random_state=42)

# Feature selection with F1 score
sfs = SequentialFeatureSelector(
    clf,
    n_features_to_select=5,
    direction='forward',
    scoring='f1',
    cv=5
)

# Fit and transform
X_selected = sfs.fit_transform(X_scaled, y)

# Display results
selected_features = sfs.get_support()
selected_feature_indices = np.where(selected_features)[0]

print("Number of selected features:", X_selected.shape[1])
print("Selected feature indices:", selected_feature_indices)

# Evaluate results on selected features using F1 score
scores = cross_val_score(clf, X_selected, y, cv=5, scoring='f1')
print("Average F1 score:", scores.mean())
print("Standard deviation:", scores.std()) 