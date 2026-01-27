import pandas as pd
import numpy as np
from ravix import parse_formula

def create_test_data():
    """Create test dataframes for different scenarios."""
    # Basic test data
    data1 = pd.DataFrame({
        'Y': [1, 2, 3, 4, 5],
        'X1': [1, 2, 3, 4, 5],
        'X2': ['A', 'B', 'A', 'B', 'A'],
        'X3': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    
    # Data for power transformation examples
    data2 = pd.DataFrame({
        'Followers': [100, 400, 900, 1600, 2500],
        'Tweets': [10, 20, 30, 40, 50]
    })
    
    # Data for interaction examples
    data3 = pd.DataFrame({
        'Likes': [10, 20, 30, 40, 50],
        'Age': [20, 30, 40, 50, 60],
        'PromoterA': [0, 1, 0, 1, 0]
    })
    
    return data1, data2, data3

def test_basic_formulas():
    """Test basic formula parsing."""
    print("=" * 70)
    print("TEST 1: Basic formulas")
    print("=" * 70)
    
    data, _, _ = create_test_data()
    
    # Test 1: Y ~ .
    print("\n1. Formula: 'Y ~ .'")
    Y, X = parse_formula("Y ~ .", data)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   Y shape: {Y.shape}")
    print(f"   X shape: {X.shape}")
    print(f"   X columns: {X.columns.tolist()}")
    assert Y.name == 'Y'
    assert 'Intercept' in X.columns
    assert 'X1' in X.columns
    print("   ✓ PASSED")
    
    # Test 2: Y ~ . - X3
    print("\n2. Formula: 'Y ~ . - X3'")
    Y, X = parse_formula("Y ~ . - X3", data)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   X columns: {X.columns.tolist()}")
    assert 'X3' not in X.columns
    assert 'X1' in X.columns
    print("   ✓ PASSED")
    
    # Test 3: Y ~ X1 + X2:X3
    print("\n3. Formula: 'Y ~ X1 + X2:X3'")
    Y, X = parse_formula("Y ~ X1 + X2:X3", data)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   X columns: {X.columns.tolist()}")
    assert 'X1' in X.columns
    assert any('X2' in p and 'X3' in p for p in X.columns)
    print("   ✓ PASSED")

def test_log_transformations():
    """Test log transformations."""
    print("\n" + "=" * 70)
    print("TEST 2: Log transformations")
    print("=" * 70)
    
    data, _, _ = create_test_data()
    
    # Test 4: log(Y) ~ log(X1) + X2
    print("\n4. Formula: 'log(Y) ~ log(X1) + X2'")
    Y, X = parse_formula("log(Y) ~ log(X1) + X2", data)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   Y (first 3): {Y.values[:3]}")
    print(f"   X columns: {X.columns.tolist()}")
    assert Y.name == 'Y'
    assert 'log(X1)' in X.columns
    assert np.allclose(Y.values[:3], np.log(data['Y'].values[:3]))
    print("   ✓ PASSED")

def test_inverse_transformations():
    """Test inverse transformations."""
    print("\n" + "=" * 70)
    print("TEST 3: Inverse transformations")
    print("=" * 70)
    
    data, _, _ = create_test_data()
    
    # Test 5: inv(Y) ~ inv(X1) + X2
    print("\n5. Formula: 'inv(Y) ~ inv(X1) + X2'")
    Y, X = parse_formula("inv(Y) ~ inv(X1) + X2", data)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   Y (first 3): {Y.values[:3]}")
    print(f"   X columns: {X.columns.tolist()}")
    assert Y.name == 'Y'
    assert 'inverse(X1)' in X.columns
    assert np.allclose(Y.values[:3], 1 / data['Y'].values[:3])
    print("   ✓ PASSED")

def test_power_transformations():
    """Test power transformations."""
    print("\n" + "=" * 70)
    print("TEST 4: Power transformations")
    print("=" * 70)
    
    data, data2, data3 = create_test_data()
    
    # Test 6: Y^2 ~ X1 + X2
    print("\n6. Formula: 'Y^2 ~ X1 + X2'")
    Y, X = parse_formula("Y^2 ~ X1 + X2", data)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   Y (first 3): {Y.values[:3]}")
    print(f"   Original Y^2 (first 3): {(data['Y']**2).values[:3]}")
    assert Y.name == 'Y'
    assert np.allclose(Y.values[:3], (data['Y']**2).values[:3])
    print("   ✓ PASSED")
    
    # Test 7: Y**2 ~ X1**2 + X2 (** same as ^)
    print("\n7. Formula: 'Y**2 ~ X1**2 + X2'")
    Y, X = parse_formula("Y**2 ~ X1**2 + X2", data)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   X columns: {X.columns.tolist()}")
    assert 'X1^2' in X.columns
    assert np.allclose(Y.values[:3], (data['Y']**2).values[:3])
    print("   ✓ PASSED")
    
    # Test 8: Followers^0.5 ~ Tweets
    print("\n8. Formula: 'Followers^0.5 ~ Tweets'")
    Y, X = parse_formula("Followers^0.5 ~ Tweets", data2)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   Y (first 3): {Y.values[:3]}")
    print(f"   Expected (first 3): {(data2['Followers']**0.5).values[:3]}")
    assert Y.name == 'Followers'
    assert np.allclose(Y.values[:3], (data2['Followers']**0.5).values[:3])
    print("   ✓ PASSED")
    
    # Test 9: Likes^2 ~ Age^2 + PromoterA:Age^2 + PromoterA
    print("\n9. Formula: 'Likes^2 ~ Age^2 + PromoterA:Age^2 + PromoterA'")
    Y, X = parse_formula("Likes^2 ~ Age^2 + PromoterA:Age^2 + PromoterA", data3)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   X columns: {X.columns.tolist()}")
    assert 'Age^2' in X.columns
    assert 'PromoterA:Age^2' in X.columns or any('Age^2' in p and 'PromoterA' in p for p in X.columns)
    assert np.allclose(Y.values[:3], (data3['Likes']**2).values[:3])
    print("   ✓ PASSED")

def test_interaction_expansion():
    """Test R-style interaction expansion."""
    print("\n" + "=" * 70)
    print("TEST 5: R-style interaction expansion")
    print("=" * 70)
    
    data, _, _ = create_test_data()
    
    # Test 10: Y ~ X1*X2 (expands to X1 + X2 + X1:X2)
    print("\n10. Formula: 'Y ~ X1*X2'")
    Y, X = parse_formula("Y ~ X1*X2", data)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   X columns: {X.columns.tolist()}")
    assert 'X1' in X.columns
    # X2 is categorical, so check for X2_B
    assert any('X2' in p for p in X.columns)
    # Check for interaction
    assert any('X1' in p and 'X2' in p and ':' in p for p in X.columns)
    print("   ✓ PASSED")

def test_no_response():
    """Test formula with no response variable."""
    print("\n" + "=" * 70)
    print("TEST 6: No response variable")
    print("=" * 70)
    
    data, _, _ = create_test_data()
    
    # Test 11: ~ X1 + X2
    print("\n11. Formula: '~ X1 + X2'")
    Y, X = parse_formula("~ X1 + X2", data)
    print(f"   Predictors: {X.columns}")
    print(f"   Y: {Y}")
    print(f"   X shape: {X.shape}")
    print(f"   X columns: {X.columns.tolist()}")
    assert Y is None
    assert 'X1' in X.columns
    assert 'X2' in X.columns
    print("   ✓ PASSED")

def test_dotpower_and_complex():
    """Test .^2 expansion and complex formulas."""
    print("\n" + "=" * 70)
    print("TEST 7: .^2 expansion and complex formulas")
    print("=" * 70)
    
    data, _, _ = create_test_data()
    
    # Test 12: Y ~ .^2 (all variables + all pairwise interactions, NO self-interactions)
    print("\n12. Formula: 'Y ~ .^2'")
    Y, X = parse_formula("Y ~ .^2", data)
    print(f"   Response: {Y.name}")
    print(f"   Number of X.columns: {len(X.columns)}")
    print(f"   Predictors: {X.columns}")
    # Should include all variables
    assert 'X1' in X.columns
    assert 'X3' in X.columns
    # Should include interactions between different variables
    assert any('X1:X3' in p for p in X.columns)
    # Should NOT include self-interactions (X1:X1, X3:X3, etc.)
    assert not any('X1:X1' in p for p in X.columns), "Should not have self-interactions"
    assert not any('X3:X3' in p for p in X.columns), "Should not have self-interactions"
    print("   ✓ PASSED")
    
    # Test 13: Y**2 ~ X1*X2 (power response with R-style expansion)
    print("\n13. Formula: 'Y**2 ~ X1*X2'")
    Y, X = parse_formula("Y**2 ~ X1*X2", data)
    print(f"   Response: {Y.name}")
    print(f"   Y (first 3): {Y.values[:3]}")
    print(f"   Expected Y^2 (first 3): {(data['Y']**2).values[:3]}")
    print(f"   Predictors: {X.columns}")
    print(f"   X columns: {X.columns.tolist()}")
    assert Y.name == 'Y'
    assert np.allclose(Y.values[:3], (data['Y']**2).values[:3])
    assert 'X1' in X.columns
    # Check for X2 dummy and interaction
    assert any('X2' in p for p in X.columns)
    assert any('X1' in p and 'X2' in p and ':' in p for p in X.columns)
    print("   ✓ PASSED")

def test_external_variables():
    """Test using variables defined outside the dataframe."""
    print("\n" + "=" * 70)
    print("TEST 8: External variables (numeric and categorical)")
    print("=" * 70)
    
    data, _, _ = create_test_data()
    # Remove X3 from data to test external variable
    data_subset = data[['Y', 'X1', 'X2']].copy()
    
    # Define external numeric variable
    Z_numeric = [10, 20, 30, 40, 50]
    
    # Test 14: External numeric variable
    print("\n14. Formula: 'Y ~ X1 + Z_numeric' (Z_numeric defined externally)")
    print(f"   Z_numeric = {Z_numeric}")
    Y, X = parse_formula("Y ~ X1 + Z_numeric", data_subset)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   X columns: {X.columns.tolist()}")
    print(f"   Z_numeric in X:\n{X['Z_numeric'].values}")
    assert 'Z_numeric' in X.columns
    assert np.allclose(X['Z_numeric'].values, Z_numeric)
    print("   ✓ PASSED")
    
    # Define external categorical variable
    Category = ['cat1', 'cat2', 'cat1', 'cat2', 'cat1']
    
    # Test 15: External categorical variable
    print("\n15. Formula: 'Y ~ X1 + Category' (Category defined externally)")
    print(f"   Category = {Category}")
    Y, X = parse_formula("Y ~ X1 + Category", data_subset)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   X columns: {X.columns.tolist()}")
    # Should create dummy variable for categorical
    assert any('Category' in p for p in X.columns)
    print("   ✓ PASSED")
    
    # Test 16: External variable in interaction
    print("\n16. Formula: 'Y ~ X1:Z_numeric' (interaction with external variable)")
    Y, X = parse_formula("Y ~ X1:Z_numeric", data_subset)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   X columns: {X.columns.tolist()}")
    assert any('X1' in p and 'Z_numeric' in p for p in X.columns)
    # Verify interaction calculation
    expected_interaction = data_subset['X1'].values * np.array(Z_numeric)
    assert np.allclose(X['X1:Z_numeric'].values, expected_interaction)
    print("   ✓ PASSED")
    
    # Test 17: External variable with transformation
    print("\n17. Formula: 'Y ~ log(Z_numeric)' (transformation of external variable)")
    Y, X = parse_formula("Y ~ log(Z_numeric)", data_subset)
    print(f"   Response: {Y.name}")
    print(f"   Predictors: {X.columns}")
    print(f"   X columns: {X.columns.tolist()}")
    assert 'log(Z_numeric)' in X.columns
    expected_log = np.log(np.array(Z_numeric))
    assert np.allclose(X['log(Z_numeric)'].values, expected_log)
    print("   ✓ PASSED")

def test_error_handling():
    """Test error handling for invalid transformations."""
    print("\n" + "=" * 70)
    print("TEST 9: Error handling")
    print("=" * 70)
    
    data, _, _ = create_test_data()
    
    # Test invalid transformation
    print("\n18. Testing invalid transformation: 'foo(Y) ~ X1'")
    try:
        Y, X = parse_formula("foo(Y) ~ X1", data)
        print("   ✗ FAILED - Should have raised ValueError")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   Expected error: {e}")
        assert "Unrecognized response transformation" in str(e)
        print("   ✓ PASSED")

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING ALL PARSE_FORMULA TESTS")
    print("=" * 70)
    
    try:
        test_basic_formulas()
        test_log_transformations()
        test_inverse_transformations()
        test_power_transformations()
        test_interaction_expansion()
        test_no_response()
        test_dotpower_and_complex()
        test_external_variables()
        test_error_handling()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()
