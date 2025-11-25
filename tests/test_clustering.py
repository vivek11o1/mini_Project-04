"""
Unit tests for src/clustering.py

Tests K-Means clustering, elbow method, silhouette scoring, and cluster assignment.
"""

import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from src.clustering import train_kmeans, calculate_elbow, calculate_silhouette, assign_clusters


class TestTrainKMeans:
    """Test suite for train_kmeans function"""

    def test_train_kmeans_basic(self):
        """Test basic K-Means training"""
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
            [10, 11]
        ])
        model = train_kmeans(features, k=2)

        # Check model attributes
        assert hasattr(model, "cluster_centers_")
        assert model.n_clusters == 2
        assert hasattr(model, "predict")

    def test_train_kmeans_with_dataframe(self):
        """Test K-Means with DataFrame input"""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "feature2": [4, 3, 2, 1]
        })
        model = train_kmeans(df, k=2)
        assert model.n_clusters == 2

    def test_train_kmeans_custom_k(self):
        """Test K-Means with custom k values"""
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
            [10, 11],
            [15, 16]
        ])
        for k in [2, 3, 4]:
            model = train_kmeans(features, k=k)
            assert model.n_clusters == k

    def test_train_kmeans_reproducibility(self):
        """Test that same random seed produces same results"""
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
            [10, 11]
        ])

        model1 = train_kmeans(features, k=2, random_state=42)
        model2 = train_kmeans(features, k=2, random_state=42)

        # Centers should be identical with same seed
        np.testing.assert_array_almost_equal(
            model1.cluster_centers_,
            model2.cluster_centers_
        )

    def test_train_kmeans_predictions(self):
        """Test that trained model can make predictions"""
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
            [10, 11]
        ])
        model = train_kmeans(features, k=2)
        predictions = model.predict(features)

        assert len(predictions) == len(features)
        assert all(p in [0, 1] for p in predictions)


class TestCalculateElbow:
    """Test suite for calculate_elbow function"""

    def test_calculate_elbow_basic(self):
        """Test basic elbow method calculation"""
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
            [10, 11]
        ])
        inertias = calculate_elbow(features, k_range=range(2, 4))

        assert len(inertias) == 2
        assert all(isinstance(i, (int, float)) for i in inertias)

    def test_calculate_elbow_decreasing(self):
        """Test that inertia decreases with more clusters"""
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
            [10, 11]
        ])
        inertias = calculate_elbow(features, k_range=range(1, 4))

        # Inertia should generally decrease
        assert inertias[0] >= inertias[1]
        assert inertias[1] >= inertias[2]

    def test_calculate_elbow_custom_range(self):
        """Test with custom k range"""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [4, 3, 2, 1, 10, 11]
        })
        inertias = calculate_elbow(df, k_range=range(2, 6))
        assert len(inertias) == 4

    def test_calculate_elbow_single_value(self):
        """Test with single k value"""
        features = np.array([[1, 2], [2, 3], [9, 10]])
        inertias = calculate_elbow(features, k_range=range(2, 3))
        assert len(inertias) == 1


class TestCalculateSilhouette:
    """Test suite for calculate_silhouette function"""

    def test_calculate_silhouette_basic(self):
        """Test basic silhouette score calculation"""
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
            [10, 11]
        ])
        score = calculate_silhouette(features, k=2)

        # Silhouette score should be between -1 and 1
        assert -1 <= score <= 1

    def test_calculate_silhouette_returns_float(self):
        """Test that silhouette returns float"""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "feature2": [4, 3, 2, 1]
        })
        score = calculate_silhouette(df, k=2)
        assert isinstance(score, (float, np.floating))

    def test_calculate_silhouette_different_k_values(self):
        """Test silhouette with different k values"""
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
            [10, 11],
            [15, 16]
        ])

        scores = []
        for k in [2, 3]:
            score = calculate_silhouette(features, k=k)
            scores.append(score)

        # Both should be valid scores
        assert all(-1 <= s <= 1 for s in scores)


class TestAssignClusters:
    """Test suite for assign_clusters function"""

    def test_assign_clusters_basic(self):
        """Test basic cluster assignment"""
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
            [10, 11]
        ])
        model = train_kmeans(features, k=2)

        df = pd.DataFrame(features, columns=['feature1', 'feature2'])
        result = assign_clusters(df, model)

        assert 'cluster' in result.columns
        assert len(result) == len(df)

    def test_assign_clusters_values(self):
        """Test that cluster values are valid"""
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
            [10, 11]
        ])
        model = train_kmeans(features, k=2)

        df = pd.DataFrame(features, columns=['feature1', 'feature2'])
        result = assign_clusters(df, model)

        assert all(c in [0, 1] for c in result['cluster'])

    def test_assign_clusters_preserves_original_data(self):
        """Test that original data is preserved"""
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10]
        ])
        model = train_kmeans(features, k=2)

        df = pd.DataFrame(features, columns=['feature1', 'feature2'])
        result = assign_clusters(df, model)

        # Original columns should still exist
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        # And have same values
        pd.testing.assert_series_equal(
            result['feature1'],
            df['feature1'],
            check_names=True
        )
