import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
from ravix.plots.hist import hist
from ravix.plots.hists import hists
from ravix.plots.hist_res import hist_res


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'y': np.random.normal(50, 10, 100),
        'x1': np.random.normal(100, 15, 100),
        'x2': np.random.normal(200, 20, 100),
        'x3': np.random.normal(300, 25, 100)
    })


@pytest.fixture
def sample_vector():
    """Generate a simple vector for testing."""
    np.random.seed(42)
    return np.random.normal(0, 1, 100)


@pytest.fixture
def mock_model():
    """Create a mock statsmodels regression model."""
    model = Mock()
    np.random.seed(42)
    model.resid = np.random.normal(0, 1, 100)
    return model


class TestHistSingleVector:
    """Test hist function with single vector input (Mode 1)."""
    
    def test_hist_with_array(self, sample_vector):
        """Test hist with numpy array."""
        hist(sample_vector, bins=20, color='green')
        plt.close('all')
    
    def test_hist_with_series(self, sample_data):
        """Test hist with pandas Series."""
        hist(sample_data['y'], bins=15, color='red')
        plt.close('all')
    
    def test_hist_with_norm(self, sample_vector):
        """Test hist with normal distribution overlay."""
        hist(sample_vector, norm=True, color='blue')
        plt.close('all')
    
    def test_hist_with_custom_labels(self, sample_vector):
        """Test hist with custom labels and title."""
        hist(sample_vector, title="Custom Title", 
             xlab="Custom X", ylab="Custom Y")
        plt.close('all')
    
    def test_hist_with_subplot(self, sample_vector):
        """Test hist with subplot specification."""
        fig = plt.figure(figsize=(10, 5))
        hist(sample_vector, subplot=(1, 2, 1))
        hist(sample_vector, subplot=(1, 2, 2))
        plt.close('all')


class TestHistFormula:
    """Test hist function with formula input (Mode 2)."""
    
    def test_hist_with_formula(self, sample_data):
        """Test hist with formula."""
        hist("y ~ x1 + x2", data=sample_data)
        plt.close('all')
    
    def test_hist_with_single_column_name(self, sample_data):
        """Test hist with single column name."""
        hist("y", data=sample_data)
        plt.close('all')
    
    def test_hist_formula_with_colors(self, sample_data):
        """Test hist with formula and custom colors."""
        hist("y ~ x1 + x2", data=sample_data, 
             xcolor='blue', ycolor='red')
        plt.close('all')
    
    def test_hist_formula_color_override(self, sample_data):
        """Test that color parameter overrides xcolor/ycolor."""
        hist("y ~ x1 + x2", data=sample_data, color='green')
        plt.close('all')
    
    def test_hist_formula_with_layout(self, sample_data):
        """Test hist with different layouts."""
        for layout in ['row', 'column', 'matrix']:
            hist("y ~ x1 + x2", data=sample_data, layout=layout)
            plt.close('all')
    
    def test_hist_formula_with_norm(self, sample_data):
        """Test hist with formula and normal distribution overlay."""
        hist("y ~ x1", data=sample_data, norm=True)
        plt.close('all')


class TestHistDataFrame:
    """Test hist function with DataFrame input (Mode 3)."""
    
    def test_hist_with_multi_column_dataframe(self, sample_data):
        """Test hist with DataFrame containing multiple numeric columns."""
        hist(sample_data)
        plt.close('all')
    
    def test_hist_with_single_column_dataframe(self, sample_data):
        """Test hist with DataFrame containing single numeric column."""
        single_col_df = sample_data[['y']]
        hist(single_col_df)
        plt.close('all')
    
    def test_hist_dataframe_with_color(self, sample_data):
        """Test hist with DataFrame and custom color."""
        hist(sample_data, color='purple')
        plt.close('all')
    
    def test_hist_dataframe_with_layout(self, sample_data):
        """Test hist with DataFrame and different layouts."""
        for layout in ['row', 'column', 'matrix']:
            hist(sample_data, layout=layout)
            plt.close('all')
    
    def test_hist_dataframe_with_mixed_types(self):
        """Test hist with DataFrame containing mixed types (should select only numeric)."""
        df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10, 20, 30, 40, 50],
            'string': ['a', 'b', 'c', 'd', 'e']
        })
        hist(df)
        plt.close('all')


class TestHistModel:
    """Test hist function with statsmodels model input (Mode 4)."""
    
    def test_hist_with_model(self, mock_model):
        """Test hist with statsmodels regression model."""
        hist(mock_model)
        plt.close('all')
    
    def test_hist_model_with_custom_labels(self, mock_model):
        """Test hist with model and custom labels."""
        hist(mock_model, title="Custom Residuals", 
             xlab="Custom X", ylab="Custom Y")
        plt.close('all')
    
    def test_hist_model_with_subplot(self, mock_model):
        """Test hist with model in subplot."""
        fig = plt.figure(figsize=(10, 5))
        hist(mock_model, subplot=(1, 2, 1))
        plt.close('all')


class TestHists:
    """Test hists function directly."""
    
    def test_hists_with_formula(self, sample_data):
        """Test hists with formula."""
        hists("y ~ x1 + x2", data=sample_data)
        plt.close('all')
    
    def test_hists_with_dataframe(self, sample_data):
        """Test hists with DataFrame."""
        hists(sample_data)
        plt.close('all')
    
    def test_hists_with_single_column(self, sample_data):
        """Test hists with single column name."""
        hists("y", data=sample_data)
        plt.close('all')
    
    def test_hists_with_colors(self, sample_data):
        """Test hists with custom colors."""
        hists("y ~ x1", data=sample_data, xcolor='green', ycolor='orange')
        plt.close('all')
    
    def test_hists_with_norm(self, sample_data):
        """Test hists with normal distribution overlay."""
        hists(sample_data, norm=True)
        plt.close('all')
    
    def test_hists_layouts(self, sample_data):
        """Test hists with different layouts."""
        for layout in ['row', 'column', 'matrix']:
            hists(sample_data, layout=layout)
            plt.close('all')
    
    def test_hists_with_custom_title(self, sample_data):
        """Test hists with custom title."""
        hists(sample_data, title="Custom Title")
        plt.close('all')
    
    def test_hists_with_infinite_values(self):
        """Test hists handles infinite values."""
        df = pd.DataFrame({
            'x1': [1, 2, np.inf, 4, 5],
            'x2': [10, -np.inf, 30, 40, 50]
        })
        hists(df)
        plt.close('all')


class TestHistRes:
    """Test hist_res function directly."""
    
    def test_hist_res_basic(self, mock_model):
        """Test hist_res with basic model."""
        hist_res(mock_model)
        plt.close('all')
    
    def test_hist_res_with_custom_title(self, mock_model):
        """Test hist_res with custom title."""
        hist_res(mock_model, title="My Residuals")
        plt.close('all')
    
    def test_hist_res_with_custom_labels(self, mock_model):
        """Test hist_res with custom axis labels."""
        hist_res(mock_model, xlab="Errors", ylab="Probability")
        plt.close('all')
    
    def test_hist_res_with_subplot(self, mock_model):
        """Test hist_res in subplot."""
        fig = plt.figure(figsize=(10, 5))
        hist_res(mock_model, subplot=(1, 2, 1))
        plt.close('all')


class TestHistParameterPassing:
    """Test that parameters are correctly passed between functions."""
    
    def test_bins_parameter(self, sample_data):
        """Test bins parameter is passed correctly."""
        hist(sample_data, bins=50)
        plt.close('all')
    
    def test_title_parameter_with_formula(self, sample_data):
        """Test title parameter with formula."""
        hist("y ~ x1", data=sample_data, title="Test Title")
        plt.close('all')
    
    def test_title_parameter_with_model(self, mock_model):
        """Test title parameter with model."""
        hist(mock_model, title="Test Residuals")
        plt.close('all')
    
    def test_xlab_ylab_parameters(self, sample_vector):
        """Test xlab and ylab parameters."""
        hist(sample_vector, xlab="X Label", ylab="Y Label")
        plt.close('all')


class TestHistEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Test hist with empty DataFrame."""
        df = pd.DataFrame()
        # Empty DataFrame with no numeric columns should be handled gracefully
        # It will have 0 numeric columns, so won't trigger multi-column mode
        # and will fail when trying to extract a single column
        try:
            hist(df)
            plt.close('all')
        except (IndexError, KeyError, ValueError):
            # Expected to fail with one of these errors
            pass
    
    def test_dataframe_no_numeric_columns(self):
        """Test hist with DataFrame containing no numeric columns."""
        df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        # DataFrame with no numeric columns will have empty numeric_cols
        # Should fail when trying to process
        try:
            hist(df)
            plt.close('all')
        except (IndexError, KeyError, ValueError):
            # Expected to fail
            pass
    
    def test_invalid_layout(self, sample_data):
        """Test hist with invalid layout parameter."""
        with pytest.raises(ValueError):
            hist(sample_data, layout='invalid')
        plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
