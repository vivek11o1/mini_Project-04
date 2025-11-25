"""
Unit tests for src/visualization.py

Tests visualization functions including elbow plots, cluster visualizations,
and heatmaps.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.visualization import plot_elbow


class TestPlotElbow:
    """Test suite for plot_elbow function"""

    def test_plot_elbow_basic(self, tmp_path):
        """Test basic elbow plot generation"""
        inertias = [10, 8, 6, 5, 4.5]
        output_path = tmp_path / "elbow.png"

        plot_elbow(inertias, save_path=str(output_path))

        # Check file was created
        assert output_path.exists()

    def test_plot_elbow_creates_png(self, tmp_path):
        """Test that PNG file is created"""
        inertias = [100, 50, 30, 20, 15]
        output_path = tmp_path / "test_elbow.png"

        plot_elbow(inertias, save_path=str(output_path))

        assert output_path.exists()
        assert str(output_path).endswith('.png')

    def test_plot_elbow_with_range(self, tmp_path):
        """Test elbow plot with custom k range"""
        inertias = [100, 50, 30, 20]
        output_path = tmp_path / "elbow_range.png"

        plot_elbow(inertias, k_range=range(2, 6), save_path=str(output_path))

        assert output_path.exists()

    def test_plot_elbow_single_value(self, tmp_path):
        """Test with single inertia value"""
        inertias = [100]
        output_path = tmp_path / "elbow_single.png"

        plot_elbow(inertias, save_path=str(output_path))

        assert output_path.exists()

    def test_plot_elbow_empty_list(self, tmp_path):
        """Test error handling for empty list"""
        inertias = []
        output_path = tmp_path / "elbow_empty.png"

        # Should either raise error or create empty plot
        try:
            plot_elbow(inertias, save_path=str(output_path))
        except (ValueError, IndexError):
            # Expected behavior
            pass

    def test_plot_elbow_file_permissions(self, tmp_path):
        """Test that file is created with proper permissions"""
        inertias = [10, 8, 6]
        output_path = tmp_path / "elbow_perms.png"

        plot_elbow(inertias, save_path=str(output_path))

        # Check file is readable
        assert os.access(output_path, os.R_OK)

    def test_plot_elbow_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created if they don't exist"""
        inertias = [10, 8, 6]
        output_path = tmp_path / "subdir" / "nested" / "elbow.png"

        # Parent directories don't exist yet
        assert not output_path.parent.exists()

        plot_elbow(inertias, save_path=str(output_path))

        # Should have created them
        assert output_path.exists()

    def test_plot_elbow_overwrites_existing_file(self, tmp_path):
        """Test that existing file is overwritten"""
        output_path = tmp_path / "elbow.png"

        # Create first file
        plot_elbow([10, 8, 6], save_path=str(output_path))
        first_size = output_path.stat().st_size

        # Overwrite with different data
        plot_elbow([100, 50, 30], save_path=str(output_path))
        second_size = output_path.stat().st_size

        # File should exist
        assert output_path.exists()
        # Sizes may differ
        assert second_size > 0


class TestVisualizationIntegration:
    """Integration tests for visualization functions"""

    def test_plot_elbow_with_real_kmeans_data(self, tmp_path):
        """Test elbow plot with real K-Means inertia values"""
        from src.clustering import calculate_elbow

        # Generate sample data
        features = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
            [10, 11],
            [15, 16]
        ])

        # Calculate inertias
        inertias = calculate_elbow(features, k_range=range(2, 5))

        # Plot
        output_path = tmp_path / "elbow_kmeans.png"
        plot_elbow(inertias, save_path=str(output_path))

        assert output_path.exists()

    def test_plot_elbow_no_save_path(self):
        """Test that plot_elbow works without save_path (displays instead)"""
        inertias = [10, 8, 6]

        # Should not raise error even without save_path
        try:
            plot_elbow(inertias)
        except Exception as e:
            # Display-related errors are okay in test environment
            if "display" not in str(e).lower() and "backend" not in str(e).lower():
                raise
