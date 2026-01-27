import warnings
warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
from ravix.plots.plot import plot
from ravix.plots.plot_res import plot_res
from ravix.plots.plot_xy import plot_xy
from ravix.plots.plots import plots


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'y': 2 + 3 * np.arange(100) + np.random.normal(0, 10, 100),
        'x': np.arange(100),
        'x1': np.random.normal(100, 15, 100),
        'x2': np.random.normal(200, 20, 100),
        'x3': np.random.normal(300, 25, 100)
    })


@pytest.fixture
def mock_linear_model():
    """Create a mock linear regression model."""
    model = Mock()
    np.random.seed(42)
    
    # Create realistic residuals and fitted values
    model.resid = np.random.normal(0, 1, 100)
    model.fittedvalues = np.linspace(10, 100, 100)
    model.__class__.__name__ = 'RegressionResultsWrapper'
    
    # Mock predict method
    model.predict = Mock(return_value=np.linspace(10, 100, 100))
    model.params = np.array([2.0, 3.0])
    
    return model


@pytest.fixture
def mock_logistic_model():
    """Create a mock logistic regression model."""
    model = Mock()
    np.random.seed(42)
    
    # Create logistic model attributes
    model.resid = np.random.normal(0, 0.5, 100)
    model.resid_pearson = np.random.normal(0, 1, 100)
    model.resid_deviance = np.random.normal(0, 1.2, 100)
    model.fittedvalues = np.random.uniform(0, 1, 100)
    model.__class__.__name__ = 'LogitResults'
    
    # Mock linear predictor
    model.linear_pred = np.random.normal(0, 2, 100)
    
    return model


@pytest.fixture
def fitted_model(sample_data):
    """Create a real fitted model for testing."""
    try:
        from ravix.modeling.fit import fit
        model = fit("y ~ x", data=sample_data)
        return model
    except:
        # If fit is not available, return a mock
        return mock_linear_model()


class TestPlotResiduals:
    """Test plot function with statsmodels model input (Mode 1)."""
    
    def test_plot_with_linear_model(self, mock_linear_model):
        """Test plot with linear regression model."""
        with patch('matplotlib.pyplot.show'):
            plot(mock_linear_model)
        plt.close('all')
    
    def test_plot_with_logistic_model(self, mock_logistic_model):
        """Test plot with logistic regression model."""
        with patch('matplotlib.pyplot.show'):
            plot(mock_logistic_model)
        plt.close('all')
    
    def test_plot_residuals_with_custom_title(self, mock_linear_model):
        """Test plot with custom title."""
        with patch('matplotlib.pyplot.show'):
            plot(mock_linear_model, title="Custom Residual Plot")
        plt.close('all')
    
    def test_plot_residuals_with_custom_labels(self, mock_linear_model):
        """Test plot with custom axis labels."""
        with patch('matplotlib.pyplot.show'):
            plot(mock_linear_model, xlab="Custom X", ylab="Custom Y")
        plt.close('all')
    
    def test_plot_residuals_types(self, mock_logistic_model):
        """Test plot with different residual types."""
        for res_type in ['resid', 'pearson', 'deviance']:
            with patch('matplotlib.pyplot.show'):
                plot(mock_logistic_model, res=res_type)
            plt.close('all')
    
    def test_plot_residuals_with_subplot(self, mock_linear_model):
        """Test plot in subplot configuration."""
        fig = plt.figure(figsize=(12, 5))
        plot(mock_linear_model, subplot=(1, 2, 1))
        plot(mock_linear_model, subplot=(1, 2, 2))
        plt.close('all')


class TestPlotScatter:
    """Test plot function with single predictor formula (Mode 2)."""
    
    def test_plot_scatter_basic(self, sample_data):
        """Test basic scatter plot."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x", data=sample_data)
        plt.close('all')
    
    def test_plot_scatter_with_model(self, sample_data, fitted_model):
        """Test scatter plot with fitted model line."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x", data=sample_data, model=fitted_model)
        plt.close('all')
    
    def test_plot_scatter_with_colors(self, sample_data):
        """Test scatter plot with custom colors."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x", data=sample_data, pcolor="green", lcolor="purple")
        plt.close('all')
    
    def test_plot_scatter_with_title(self, sample_data):
        """Test scatter plot with custom title."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x", data=sample_data, title="Custom Scatter Plot")
        plt.close('all')
    
    def test_plot_scatter_with_labels(self, sample_data):
        """Test scatter plot with custom labels."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x", data=sample_data, xlab="X Variable", ylab="Y Variable")
        plt.close('all')
    
    def test_plot_scatter_with_alpha(self, sample_data):
        """Test scatter plot with transparency."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x", data=sample_data, alpha=0.5)
        plt.close('all')
    
    def test_plot_scatter_with_point_size(self, sample_data):
        """Test scatter plot with custom point size."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x", data=sample_data, psize=100)
        plt.close('all')
    
    def test_plot_scatter_multiple_models(self, sample_data, fitted_model):
        """Test scatter plot with multiple model lines."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x", data=sample_data, model=[fitted_model, "line"], 
                 lcolor=["red", "blue"], legend_labels=["Model 1", "Model 2"])
        plt.close('all')
    
    def test_plot_scatter_with_subplot(self, sample_data):
        """Test scatter plot in subplot."""
        fig = plt.figure(figsize=(12, 5))
        plot("y ~ x", data=sample_data, subplot=(1, 2, 1))
        plt.close('all')


class TestPlotMatrix:
    """Test plot function with multiple predictors (Mode 3)."""
    
    def test_plot_matrix_basic(self, sample_data):
        """Test basic scatter plot matrix."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x1 + x2", data=sample_data)
        plt.close('all')
    
    def test_plot_matrix_with_colors(self, sample_data):
        """Test scatter plot matrix with custom colors."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x1 + x2 + x3", data=sample_data, 
                 xcolor="green", ycolor="orange")
        plt.close('all')
    
    def test_plot_matrix_with_lines(self, sample_data):
        """Test scatter plot matrix with regression lines."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x1 + x2", data=sample_data, lines=True, linescolor="red")
        plt.close('all')
    
    def test_plot_matrix_with_title(self, sample_data):
        """Test scatter plot matrix with custom title."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x1 + x2", data=sample_data, title="Custom Matrix Plot")
        plt.close('all')


class TestPlotRes:
    """Test plot_res function directly."""
    
    def test_plot_res_basic(self, mock_linear_model):
        """Test basic residual plot."""
        with patch('matplotlib.pyplot.show'):
            plot_res(mock_linear_model)
        plt.close('all')
    
    def test_plot_res_with_title(self, mock_linear_model):
        """Test residual plot with custom title."""
        with patch('matplotlib.pyplot.show'):
            plot_res(mock_linear_model, title="My Residuals")
        plt.close('all')
    
    def test_plot_res_residual_types(self, mock_logistic_model):
        """Test different residual types."""
        for res_type in ['resid', 'pearson', 'deviance']:
            with patch('matplotlib.pyplot.show'):
                plot_res(mock_logistic_model, res=res_type)
            plt.close('all')
    
    def test_plot_res_with_labels(self, mock_linear_model):
        """Test residual plot with custom labels."""
        with patch('matplotlib.pyplot.show'):
            plot_res(mock_linear_model, xlab="Fitted", ylab="Resids")
        plt.close('all')
    
    def test_plot_res_subplot(self, mock_linear_model):
        """Test residual plot in subplot."""
        fig = plt.figure(figsize=(10, 5))
        plot_res(mock_linear_model, subplot=(1, 2, 1))
        plt.close('all')
    
    def test_plot_res_invalid_residual_type(self, mock_linear_model):
        """Test error handling for invalid residual type."""
        with pytest.raises(ValueError):
            plot_res(mock_linear_model, res="invalid_type")
        plt.close('all')


class TestPlotXY:
    """Test plot_xy function directly."""
    
    def test_plot_xy_basic(self, sample_data):
        """Test basic xy scatter plot."""
        with patch('matplotlib.pyplot.show'):
            plot_xy("y ~ x", data=sample_data)
        plt.close('all')
    
    def test_plot_xy_with_model(self, sample_data, fitted_model):
        """Test xy plot with fitted model."""
        with patch('matplotlib.pyplot.show'):
            plot_xy("y ~ x", data=sample_data, model=fitted_model)
        plt.close('all')
    
    def test_plot_xy_with_colors(self, sample_data):
        """Test xy plot with custom colors."""
        with patch('matplotlib.pyplot.show'):
            plot_xy("y ~ x", data=sample_data, pcolor="red", lcolor="blue")
        plt.close('all')
    
    def test_plot_xy_with_title_and_labels(self, sample_data):
        """Test xy plot with custom title and labels."""
        with patch('matplotlib.pyplot.show'):
            plot_xy("y ~ x", data=sample_data, title="Custom", 
                   xlab="X Axis", ylab="Y Axis")
        plt.close('all')
    
    def test_plot_xy_with_alpha(self, sample_data):
        """Test xy plot with transparency."""
        with patch('matplotlib.pyplot.show'):
            plot_xy("y ~ x", data=sample_data, alpha=0.3)
        plt.close('all')
    
    def test_plot_xy_multiple_models(self, sample_data, fitted_model):
        """Test xy plot with multiple models."""
        with patch('matplotlib.pyplot.show'):
            plot_xy("y ~ x", data=sample_data, model=[fitted_model, "line"],
                   lcolor=["red", "green"])
        plt.close('all')
    
    def test_plot_xy_colored_points(self, sample_data):
        """Test xy plot with colored points array."""
        colors = ['red' if i % 2 == 0 else 'blue' for i in range(len(sample_data))]
        with patch('matplotlib.pyplot.show'):
            plot_xy("y ~ x", data=sample_data, pcolor=colors)
        plt.close('all')
    
    def test_plot_xy_subplot(self, sample_data):
        """Test xy plot in subplot."""
        plot_xy("y ~ x", data=sample_data, subplot=(1, 3, 1))
        plot_xy("y ~ x1", data=sample_data, subplot=(1, 3, 2))
        plot_xy("y ~ x2", data=sample_data, subplot=(1, 3, 3))
        plt.close('all')


class TestPlots:
    """Test plots function directly."""
    
    def test_plots_basic(self, sample_data):
        """Test basic scatter plot matrix."""
        with patch('matplotlib.pyplot.show'):
            plots("y ~ x1 + x2", data=sample_data)
        plt.close('all')
    
    def test_plots_with_colors(self, sample_data):
        """Test scatter matrix with custom colors."""
        with patch('matplotlib.pyplot.show'):
            plots("y ~ x1 + x2 + x3", data=sample_data, 
                  xcolor="purple", ycolor="orange")
        plt.close('all')
    
    def test_plots_with_lines(self, sample_data):
        """Test scatter matrix with regression lines."""
        with patch('matplotlib.pyplot.show'):
            plots("y ~ x1 + x2", data=sample_data, lines=True)
        plt.close('all')
    
    def test_plots_with_custom_line_color(self, sample_data):
        """Test scatter matrix with custom line color."""
        with patch('matplotlib.pyplot.show'):
            plots("y ~ x1 + x2", data=sample_data, lines=True, linescolor="green")
        plt.close('all')
    
    def test_plots_with_title(self, sample_data):
        """Test scatter matrix with custom title."""
        with patch('matplotlib.pyplot.show'):
            plots("y ~ x1 + x2", data=sample_data, title="My Matrix")
        plt.close('all')
    
    def test_plots_single_predictor_warning(self, sample_data, capsys):
        """Test that plots warns when given single predictor."""
        with patch('matplotlib.pyplot.show'):
            plots("y ~ x", data=sample_data)
        captured = capsys.readouterr()
        assert "multiple predictor" in captured.out
        plt.close('all')


class TestPlotExplicitFormula:
    """Test plot function with explicit formula parameter."""
    
    def test_plot_explicit_formula_scatter(self, sample_data):
        """Test plot with explicit formula parameter for scatter."""
        with patch('matplotlib.pyplot.show'):
            plot(None, data=sample_data, formula="y ~ x")
        plt.close('all')
    
    def test_plot_explicit_formula_matrix(self, sample_data):
        """Test plot with explicit formula parameter for matrix."""
        with patch('matplotlib.pyplot.show'):
            plot(None, data=sample_data, formula="y ~ x1 + x2")
        plt.close('all')
    
    def test_plot_formula_overrides_input_data(self, sample_data):
        """Test that formula parameter overrides input_data."""
        with patch('matplotlib.pyplot.show'):
            # Even though input_data has multiple predictors, formula overrides
            plot("y ~ x1 + x2 + x3", data=sample_data, formula="y ~ x")
        plt.close('all')


class TestPlotParameterPassing:
    """Test that parameters are correctly passed between functions."""
    
    def test_title_parameter_residuals(self, mock_linear_model):
        """Test title parameter with residuals."""
        with patch('matplotlib.pyplot.show'):
            plot(mock_linear_model, title="Test Title")
        plt.close('all')
    
    def test_title_parameter_scatter(self, sample_data):
        """Test title parameter with scatter plot."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x", data=sample_data, title="Test Scatter")
        plt.close('all')
    
    def test_title_parameter_matrix(self, sample_data):
        """Test title parameter with matrix plot."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x1 + x2", data=sample_data, title="Test Matrix")
        plt.close('all')
    
    def test_labels_scatter(self, sample_data):
        """Test xlab and ylab with scatter plot."""
        with patch('matplotlib.pyplot.show'):
            plot("y ~ x", data=sample_data, xlab="X Label", ylab="Y Label")
        plt.close('all')
    
    def test_labels_residuals(self, mock_linear_model):
        """Test xlab and ylab with residuals."""
        with patch('matplotlib.pyplot.show'):
            plot(mock_linear_model, xlab="Fitted", ylab="Resids")
        plt.close('all')


class TestPlotEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_plot_invalid_input(self):
        """Test plot with invalid input."""
        with pytest.raises(ValueError):
            plot("invalid")
        plt.close('all')
    
    def test_plot_no_data(self):
        """Test plot with formula but no data."""
        with pytest.raises(Exception):
            plot("y ~ x")
        plt.close('all')
    
    def test_plot_empty_dataframe(self):
        """Test plot with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(Exception):
            plot("y ~ x", data=df)
        plt.close('all')
    
    def test_plot_missing_columns(self, sample_data):
        """Test plot with non-existent columns."""
        with pytest.raises(Exception):
            plot("y ~ nonexistent", data=sample_data)
        plt.close('all')


class TestPlotAutomaticTitles:
    """Test automatic title generation."""
    
    def test_auto_title_scatter(self, sample_data):
        """Test automatic title for scatter plot."""
        with patch('matplotlib.pyplot.show'):
            # Should auto-generate title like "y vs x"
            plot("y ~ x", data=sample_data)
        plt.close('all')
    
    def test_auto_title_matrix(self, sample_data):
        """Test automatic title for matrix plot."""
        with patch('matplotlib.pyplot.show'):
            # Should auto-generate title like "Scatter Plot Matrix"
            plot("y ~ x1 + x2", data=sample_data)
        plt.close('all')
    
    def test_auto_title_residuals(self, mock_linear_model):
        """Test automatic title for residuals."""
        with patch('matplotlib.pyplot.show'):
            # Should auto-generate title like "Residual Plot"
            plot(mock_linear_model)
        plt.close('all')


class TestPlotIntegration:
    """Integration tests with real models."""
    
    def test_full_workflow_linear(self, sample_data):
        """Test complete workflow: fit model, plot scatter, plot residuals."""
        try:
            from ravix.modeling.fit import fit
            
            # Fit model
            model = fit("y ~ x", data=sample_data)
            
            # Plot scatter with model
            with patch('matplotlib.pyplot.show'):
                plot("y ~ x", data=sample_data, model=model)
            
            # Plot residuals
            with patch('matplotlib.pyplot.show'):
                plot(model)
            
            plt.close('all')
        except ImportError:
            pytest.skip("fit function not available")
    
    def test_multiple_plots_subplot(self, sample_data):
        """Test multiple plots in subplot configuration."""
        try:
            from ravix.modeling.fit import fit
            model = fit("y ~ x", data=sample_data)
            
            fig = plt.figure(figsize=(15, 5))
            
            # Scatter plot
            plot("y ~ x", data=sample_data, model=model, subplot=(1, 3, 1))
            
            # Residuals
            plot(model, subplot=(1, 3, 2))
            
            # Different scatter
            plot("y ~ x1", data=sample_data, subplot=(1, 3, 3))
            
            plt.close('all')
        except ImportError:
            pytest.skip("fit function not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
