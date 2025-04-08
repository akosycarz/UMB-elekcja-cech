import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

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

# Get selected features
selected_features = sfs.get_support()
selected_feature_indices = np.where(selected_features)[0]

# 1. Feature Importance Plot
plt.figure(figsize=(12, 6))
feature_importance = np.zeros(X.shape[1])
feature_importance[selected_feature_indices] = 1
plt.bar(range(X.shape[1]), feature_importance)
plt.title('Selected Features')
plt.xlabel('Feature Index')
plt.ylabel('Selected (1) / Not Selected (0)')
plt.xticks(range(X.shape[1]))
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# 2. Cross-validation Scores Distribution
scores = cross_val_score(clf, X_selected, y, cv=5, scoring='f1')
plt.figure(figsize=(10, 6))
sns.boxplot(x=scores)
plt.title('Distribution of F1 Scores Across Cross-validation Folds')
plt.xlabel('F1 Score')
plt.tight_layout()
plt.savefig('cv_scores_distribution.png')
plt.show()

# 3. Feature Correlation Heatmap
plt.figure(figsize=(12, 10))
selected_features_data = X_scaled[:, selected_feature_indices]
correlation_matrix = np.corrcoef(selected_features_data.T)
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm',
            xticklabels=selected_feature_indices,
            yticklabels=selected_feature_indices)
plt.title('Correlation Matrix of Selected Features')
plt.tight_layout()
plt.savefig('feature_correlation.png')
plt.show()

# 4. Feature Selection Progress
n_features_range = range(1, len(selected_feature_indices) + 1)
scores_progress = []
for n in n_features_range:
    sfs_n = SequentialFeatureSelector(
        clf,
        n_features_to_select=n,
        direction='forward',
        scoring='f1',
        cv=5
    )
    X_selected_n = sfs_n.fit_transform(X_scaled, y)
    scores_n = cross_val_score(clf, X_selected_n, y, cv=5, scoring='f1')
    scores_progress.append(scores_n.mean())

plt.figure(figsize=(10, 6))
plt.plot(n_features_range, scores_progress, marker='o')
plt.title('F1 Score vs Number of Selected Features')
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.grid(True)
plt.tight_layout()
plt.savefig('feature_selection_progress.png')
plt.show() 