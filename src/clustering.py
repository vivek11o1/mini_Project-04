import pickle
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def train_kmeans(features, k: int = 4, random_state: int = 42):
    """Train and persist a K-Means model."""
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    model.fit(features)
    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/kmeans_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model


def calculate_elbow(features, k_range=range(2, 11)):
    """Return inertia values across k_range for elbow plotting."""
    inertias = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        model.fit(features)
        inertias.append(model.inertia_)
    return inertias


def calculate_silhouette(features, k: int):
    """Compute silhouette score for a given k."""
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(features)
    return silhouette_score(features, labels)


def assign_clusters(df, model):
    """Append cluster labels to the original dataframe."""
    labeled = df.copy()
    labeled["cluster"] = model.predict(df)
    return labeled
