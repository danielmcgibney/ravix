import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ravix import barplot


class TestBarplot(unittest.TestCase):
    """Test suite for the barplot function"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests"""
        np.random.seed(42)
        
        # Dataset 1: Simple numeric data
        cls.simple_df = pd.DataFrame({
            'temperature': [72, 75, 68, 80, 77, 71, 73],
            'humidity': [45, 50, 42, 55, 48, 44, 46],
            'pressure': [1013, 1015, 1012, 1018, 1014, 1013, 1016]
        })
        
        # Dataset 2: Cars data with numeric variables
        cls.cars_df = pd.DataFrame({
            'price': [25000, 30000, 22000, 35000, 28000, 32000, 26000, 29000],
            'age': [3, 2, 5, 1, 4, 2, 3, 2],
            'mileage': [45000, 30000, 75000, 15000, 60000, 25000, 50000, 35000]
        })
        
        # Dataset 3: Employee data with categorical variable
        cls.employees_df = pd.DataFrame({
            'salary': [75000, 82000, 68000, 95000, 78000, 88000, 72000, 85000, 
                       90000, 70000, 77000, 83000, 92000, 86000, 74000],
            'department': ['Sales', 'Sales', 'Sales', 'Engineering', 'Engineering', 
                           'Engineering', 'HR', 'HR', 'Engineering', 'Sales', 
                           'HR', 'Engineering', 'Engineering', 'Sales', 'HR'],
            'years_experience': [5, 7, 3, 10, 6, 9, 4, 8, 12, 4, 5, 8, 11, 7, 3]
        })
        
        # Dataset 4: Pre-aggregated data with index
        cls.dept_summary = pd.DataFrame({
            'department': ['Sales', 'Engineering', 'HR'],
            'avg_salary': [73750, 87000, 75000]
        })
        cls.dept_summary_indexed = cls.dept_summary.set_index('department')
        
        # Dataset 5: NumPy array
        cls.dept_array = cls.dept_summary_indexed.to_numpy()
        
        # Close all plots after each test to avoid display issues
        plt.ioff()
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    # Test Case 1: DataFrame with mean aggregation
    def test_dataframe_mean_aggregation(self):
        """Test barplot with DataFrame and mean aggregation"""
        try:
            barplot(self.simple_df, agg='mean', title='Test: Mean Aggregation')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"DataFrame mean aggregation failed: {e}")
    
    # Test Case 2: DataFrame with sum aggregation
    def test_dataframe_sum_aggregation(self):
        """Test barplot with DataFrame and sum aggregation"""
        try:
            barplot(self.simple_df, agg='sum', title='Test: Sum Aggregation')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"DataFrame sum aggregation failed: {e}")
    
    # Test Case 3: Formula with numeric variables (mean)
    def test_formula_numeric_mean(self):
        """Test barplot with formula and numeric variables (mean)"""
        try:
            barplot('price ~ age + mileage', data=self.cars_df, agg='mean',
                   title='Test: Formula with Mean')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Formula with numeric variables (mean) failed: {e}")
    
    # Test Case 4: Formula with numeric variables (median)
    def test_formula_numeric_median(self):
        """Test barplot with formula and numeric variables (median)"""
        try:
            barplot('price ~ age + mileage', data=self.cars_df, agg='median',
                   title='Test: Formula with Median')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Formula with numeric variables (median) failed: {e}")
    
    # Test Case 5: Y ~ categorical with mean aggregation
    def test_categorical_mean(self):
        """Test barplot with Y ~ categorical and mean aggregation"""
        try:
            barplot('salary ~ department', data=self.employees_df, agg='mean',
                   title='Test: Categorical Mean')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Y ~ categorical with mean failed: {e}")
    
    # Test Case 6: Y ~ categorical with count aggregation
    def test_categorical_count(self):
        """Test barplot with Y ~ categorical and count aggregation"""
        try:
            barplot('salary ~ department', data=self.employees_df, agg='count',
                   title='Test: Categorical Count')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Y ~ categorical with count failed: {e}")
    
    # Test Case 7: Pre-aggregated data with formula (agg=None)
    def test_preaggregated_formula(self):
        """Test barplot with pre-aggregated data and formula"""
        try:
            barplot('avg_salary ~ department', data=self.dept_summary, agg=None,
                   title='Test: Pre-aggregated Formula')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Pre-aggregated data with formula failed: {e}")
    
    # Test Case 8: Single row DataFrame (agg=None)
    def test_single_row_dataframe(self):
        """Test barplot with single row DataFrame"""
        summary_stats = pd.DataFrame({
            'Mean': [75.5],
            'Median': [73.0],
            'Max': [80.0]
        })
        try:
            barplot(summary_stats, agg=None, title='Test: Single Row')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Single row DataFrame failed: {e}")
    
    # Test Case 9: DataFrame with index as labels (agg=None)
    def test_dataframe_with_index(self):
        """Test barplot with DataFrame using index as labels"""
        try:
            barplot(self.dept_summary_indexed, agg=None,
                   title='Test: Index as Labels')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"DataFrame with index failed: {e}")
    
    # Test Case 10: NumPy array input
    def test_numpy_array(self):
        """Test barplot with NumPy array input"""
        try:
            barplot(self.dept_array, agg=None, title='Test: NumPy Array')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"NumPy array input failed: {e}")
    
    # Test Case 11: Horizontal bar plot
    def test_horizontal_barplot(self):
        """Test barplot with horizontal orientation"""
        try:
            barplot('salary ~ department', data=self.employees_df, agg='mean',
                   horizontal=True, title='Test: Horizontal')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Horizontal bar plot failed: {e}")
    
    # Test Case 12: Single color for all bars
    def test_single_color(self):
        """Test barplot with single color"""
        try:
            barplot(self.simple_df, agg='mean', color='green',
                   title='Test: Single Color')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Single color failed: {e}")
    
    # Test Case 13: List of colors for each bar
    def test_color_list(self):
        """Test barplot with list of colors"""
        try:
            barplot(self.dept_summary_indexed, agg=None, 
                   color=['red', 'blue', 'orange'],
                   title='Test: Color List')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"List of colors failed: {e}")
    
    # Test Case 14: List of colors with categorical formula
    def test_color_list_categorical(self):
        """Test barplot with list of hex colors and categorical formula"""
        try:
            barplot('salary ~ department', data=self.employees_df, agg='mean',
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                   title='Test: Hex Colors Categorical')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"List of hex colors with categorical failed: {e}")
    
    # Test Case 15: Invalid aggregation method
    def test_invalid_aggregation(self):
        """Test that invalid aggregation method raises ValueError"""
        with self.assertRaises(ValueError):
            barplot(self.simple_df, agg='invalid')
    
    # Test Case 16: Y ~ categorical with multiple values per category and agg=None
    def test_categorical_no_agg_error(self):
        """Test that Y ~ categorical with multiple values and agg=None raises error"""
        with self.assertRaises(ValueError):
            barplot('salary ~ department', data=self.employees_df, agg=None)
    
    # Test Case 17: Median aggregation
    def test_median_aggregation(self):
        """Test barplot with median aggregation"""
        try:
            barplot(self.simple_df, agg='median', title='Test: Median')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Median aggregation failed: {e}")
    
    # Test Case 18: Count aggregation
    def test_count_aggregation(self):
        """Test barplot with count aggregation"""
        try:
            barplot(self.simple_df, agg='count', title='Test: Count')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Count aggregation failed: {e}")
    
    # Test Case 19: Custom xcolor and ycolor
    def test_custom_xy_colors(self):
        """Test barplot with custom xcolor and ycolor"""
        try:
            barplot('price ~ age + mileage', data=self.cars_df, agg='mean',
                   xcolor='purple', ycolor='orange',
                   title='Test: Custom X/Y Colors')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Custom xcolor/ycolor failed: {e}")
    
    # Test Case 20: Subplot usage
    def test_subplot(self):
        """Test barplot with subplot parameter"""
        try:
            plt.figure(figsize=(12, 4))
            barplot(self.simple_df, agg='mean', subplot=(1, 2, 1), 
                   title='Test: Subplot 1')
            barplot(self.cars_df, agg='median', subplot=(1, 2, 2),
                   title='Test: Subplot 2')
            plt.tight_layout()
            plt.close()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Subplot usage failed: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
