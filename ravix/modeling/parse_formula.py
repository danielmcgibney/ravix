import numpy as np
import pandas as pd
import re
import inspect
from pandas.api.types import CategoricalDtype
from itertools import combinations, product


def parse_formula(formula, data=None, drop_first=True):
    """
    Parse a formula string and return Y and X for regression modeling.
    
    Args:
        formula (str): Formula string like "Y ~ X1 + X2", "~ X1 + X2" (no response), or "Y" (no predictors)
        data (pd.DataFrame): DataFrame containing the variables
        drop_first (bool): Whether to drop first level of categorical variables
        
    Returns:
        tuple: (Y, X) where:
            - Y is a pandas Series (or None if no response specified)
            - X is a pandas DataFrame (or None if no predictors specified)
    
    Examples:
        >>> data = pd.DataFrame({
        ...     'Y': [1, 2, 3, 4],
        ...     'X1': [1, 2, 3, 4],
        ...     'X2': ['A', 'B', 'A', 'B'],
        ...     'X3': [0.1, 0.2, 0.3, 0.4]
        ... })
        >>> Y, X = parse_formula("Y ~ .", data)
        >>> Y, X = parse_formula("Y ~ . - X3", data)
        >>> Y, X = parse_formula("Y ~ X1 + X2:X3", data)
        >>> Y, X = parse_formula("log(Y) ~ log(X1) + X2", data)
        >>> Y, X = parse_formula("inv(Y) ~ inv(X1) + X2", data)
        >>> Y, X = parse_formula("Y^2 ~ X1 + X2", data)
        >>> Y, X = parse_formula("Y**2 ~ X1**2 + X2", data)  # ** same as ^
        >>> Y, X = parse_formula("Followers^0.5 ~ Tweets", data)
        >>> Y, X = parse_formula("Likes^2 ~ Age^2 + PromoterA:Age^2 + PromoterA", data)
        >>> Y, X = parse_formula("Y ~ X1*X2", data)  # X1 + X2 + X1:X2
        >>> Y, X = parse_formula("~ X1 + X2", data)  # No response variable (Y is None)
        >>> Y, X = parse_formula("Y", data)  # No predictors (X is None)
    """
    if data is None:
        data = pd.DataFrame()
    
    data = data.copy()
    
    # Replace ** with ^ before processing
    formula = formula.replace('**', '^')
    formula = formula.replace(' ', '')
    
    # Check if tilde exists
    if '~' not in formula:
        # No tilde: formula specifies only Y, X is None
        response_var, resp_func, resp_power = _parse_response(formula)
        
        # Resolve external variables
        _resolve_external_variable(response_var, data)
        
        # Apply response transformation
        Y = _apply_response_transformation(response_var, resp_func, resp_power, data)
        X = None
        
        return Y, X
        
    # Tilde exists: parse normally
    response_part, predictors_part = formula.split('~', 1)
    
    # Handle blank response variable
    if response_part.strip() == '':
        Y = None
        response_var = None
    else:
        # Parse response variable (handle transformations including powers)
        response_var, resp_func, resp_power = _parse_response(response_part)
        
        # Resolve external variables
        _resolve_external_variable(response_var, data)
        
        # Apply response transformation
        Y = _apply_response_transformation(response_var, resp_func, resp_power, data)
    
    # Parse predictors
    include_intercept, predictor_terms = _parse_predictors(predictors_part, data, response_var)
    
    # Build design matrix
    X = _build_design_matrix(predictor_terms, include_intercept, data, response_var, drop_first)
    
    return Y, X


def _parse_response(response_part):
    """Parse response variable, handling transformations including power transformations."""
    response_part = response_part.strip()
    
    # Check for power transformations first (e.g., "Y^2", "Y^0.5", "Y^(2)")
    power_match = re.match(r'^(\w+)\^(\(?([\d\.]+)\)?)$', response_part)
    if power_match:
        var_name = power_match.group(1)
        power_str = power_match.group(3) if power_match.group(3) else power_match.group(2)
        power = float(power_str)
        return var_name, 'power', power
    
    # Check for function transformations (e.g., "log(Y)", "sqrt(Y)", "inv(Y)")
    func_match = re.match(r'^(\w+)\((\w+)\)$', response_part)
    if func_match:
        func_name, var_name = func_match.groups()
        # Normalize 'inv' to 'inverse'
        if func_name == 'inv':
            func_name = 'inverse'
        # Validate function name
        valid_functions = ['log', 'sqrt', 'inverse']
        if func_name not in valid_functions:
            raise ValueError(f"Unrecognized response transformation: '{func_name}'. Valid options are: {', '.join(valid_functions)} (or 'inv' for inverse)")
        return var_name, func_name, None
    
    # Simple variable name
    return response_part, None, None


def _parse_predictors(predictors_part, data, response_var):
    """Parse predictor part of formula."""
    predictors_part = predictors_part.strip()
    
    # Check for intercept exclusion
    include_intercept = True
    if '+0' in predictors_part or '-1' in predictors_part:
        include_intercept = False
        predictors_part = predictors_part.replace('+0', '').replace('-1', '')
    
    # Handle .^2 expansion (all variables + all pairwise interactions, NO self-interactions)
    if '.^2' in predictors_part:
        all_vars = [col for col in data.columns if col != response_var]
        # Add all variables
        expanded_terms = all_vars.copy()
        # Add all pairwise interactions (but not self-interactions)
        for i in range(len(all_vars)):
            for j in range(i + 1, len(all_vars)):
                expanded_terms.append(f'{all_vars[i]}:{all_vars[j]}')
        predictors_part = '+'.join(expanded_terms)
    
    # Tokenize into terms
    terms = _tokenize_formula(predictors_part)
    
    # Separate positive and negative terms
    positive_terms = []
    negative_terms = []
    
    for term in terms:
        if term.startswith('-'):
            negative_terms.append(term[1:])
        else:
            positive_terms.append(term)
    
    # Expand R-style * operator and dot notation in positive terms
    expanded_positive = []
    for term in positive_terms:
        if '*' in term:
            # R-style expansion: A*B means A + B + A:B
            expanded_positive.extend(_expand_star_notation(term))
        elif '.' in term:
            expanded_positive.extend(_expand_dot_notation(term, data, response_var))
        else:
            expanded_positive.append(term)
    
    # Expand dot notation in negative terms (for proper exclusion)
    expanded_negative = []
    for term in negative_terms:
        if '*' in term:
            # Expand * in negative terms too
            expanded_negative.extend(_expand_star_notation(term))
        elif '.' in term:
            expanded_negative.extend(_expand_dot_notation(term, data, response_var))
        else:
            expanded_negative.append(term)
    
    # Remove negative terms from positive terms
    final_terms = []
    for term in expanded_positive:
        if term not in expanded_negative:
            final_terms.append(term)
    
    return include_intercept, final_terms


def _expand_star_notation(term):
    """
    Expand R-style * notation. A*B means A + B + A:B
    A*B*C means A + B + C + A:B + A:C + B:C + A:B:C
    """
    # Split on * but preserve power operators (^)
    # We need to handle cases like A^2*B where we don't want to split A^2
    parts = term.split('*')
    
    expanded = []
    
    # Add all individual terms
    expanded.extend(parts)
    
    # Add all interactions (2-way, 3-way, etc.)
    for r in range(2, len(parts) + 1):
        for combo in combinations(parts, r):
            expanded.append(':'.join(combo))
    
    return expanded


def _tokenize_formula(formula_part):
    """Tokenize formula respecting parentheses and operators."""
    tokens = []
    current_token = ''
    paren_depth = 0
    
    i = 0
    while i < len(formula_part):
        char = formula_part[i]
        
        if char == '(':
            paren_depth += 1
            current_token += char
        elif char == ')':
            paren_depth -= 1
            current_token += char
        elif char == '+' and paren_depth == 0:
            if current_token.strip():
                tokens.append(current_token.strip())
            current_token = ''
        elif char == '-' and paren_depth == 0:
            # Check if this is a negative term or subtraction
            if current_token.strip():
                tokens.append(current_token.strip())
                current_token = '-'
            else:
                current_token = '-'
        else:
            current_token += char
        
        i += 1
    
    if current_token.strip():
        tokens.append(current_token.strip())
    
    return tokens


def _expand_dot_notation(term, data, response_var):
    """Expand dot notation to actual variable names."""
    available_vars = [col for col in data.columns if col != response_var]
    
    if term == '.':
        return available_vars
    elif ':' in term:
        # Handle interactions with dot
        parts = term.split(':')
        expanded_parts = []
        
        for part in parts:
            if part == '.':
                expanded_parts.append(available_vars)
            else:
                expanded_parts.append([part])
        
        # Generate all combinations
        interactions = []
        for combo in product(*expanded_parts):
            if len(set(combo)) == len(combo):  # No self-interactions
                interactions.append(':'.join(combo))
        
        return interactions
    else:
        return [term]


def _build_design_matrix(terms, include_intercept, data, response_var, drop_first):
    """Build the design matrix from parsed terms."""
    X_dict = {}
    
    for term in terms:
        term_matrix = _evaluate_term(term, data, response_var, drop_first)
        X_dict.update(term_matrix)
    
    X = pd.DataFrame(X_dict, index=data.index)
    
    if include_intercept:
        X.insert(0, 'Intercept', 1.0)
    
    return X


def _evaluate_term(term, data, response_var, drop_first):
    """Evaluate a single term and return dictionary of columns."""
    term = term.strip()
    
    # Handle interactions first, then power transformations within each part
    if ':' in term:
        return _handle_interaction(term, data, response_var, drop_first)
    
    # Handle power transformations for single variables
    # Updated regex to handle both X^2 and X^(2) formats
    power_match = re.match(r'^(\w+)\^(\(?([\d\.]+)\)?)$', term)
    if power_match:
        var_name = power_match.group(1)
        power_str = power_match.group(3) if power_match.group(3) else power_match.group(2)
        power = float(power_str)
        
        # Format power nicely (avoid .0 for integers)
        if power == int(power):
            power_display = str(int(power))
        else:
            power_display = str(power)
        
        _resolve_external_variable(var_name, data)
        base_matrix = _handle_variable(var_name, data, drop_first)
        
        result = {}
        for col_name, col_values in base_matrix.items():
            # Validate for negative powers (division by zero)
            if power < 0 and (col_values == 0).any():
                invalid_indices = col_values[col_values == 0].index.tolist()
                n_invalid = len(invalid_indices)
                sample_indices = invalid_indices[:5]
                raise ValueError(
                    f"Cannot compute {col_name}^{power}: found {n_invalid} zero value(s).\n"
                    f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                    f"Suggestions:\n"
                    f"  - Filter out zero values from your data\n"
                    f"  - Add a small constant: ({col_name} + c)^{power}\n"
                    f"  - Use a different transformation"
                )
            # Validate for fractional powers (roots of negative numbers)
            elif 0 < power < 1 and (col_values < 0).any():
                invalid_indices = col_values[col_values < 0].index.tolist()
                n_invalid = len(invalid_indices)
                sample_indices = invalid_indices[:5]
                raise ValueError(
                    f"Cannot compute {col_name}^{power}: found {n_invalid} negative value(s).\n"
                    f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                    f"Suggestions:\n"
                    f"  - Filter out negative values from your data\n"
                    f"  - Take absolute value if appropriate\n"
                    f"  - Use a different transformation"
                )
            
            transformed = col_values ** power
            result[f'{col_name}^{power_display}'] = transformed
        
        return result
    
    # Handle function calls
    func_match = re.match(r'^(\w+)\((.+)\)$', term)
    if func_match:
        func_name, inner_term = func_match.groups()
        return _apply_function(func_name, inner_term, data, response_var, drop_first)
    
    # Handle simple variable
    return _handle_variable(term, data, drop_first)


def _handle_interaction(term, data, response_var, drop_first):
    """Handle interaction terms like A:B or A:B:C, including power transformations."""
    parts = term.split(':')
    
    # Evaluate each part (which may include power transformations)
    part_matrices = []
    for part in parts:
        # Handle power transformations within interaction parts
        power_match = re.match(r'^(\w+)\^(\(?([\d\.]+)\)?)$', part)
        if power_match:
            var_name = power_match.group(1)
            power_str = power_match.group(3) if power_match.group(3) else power_match.group(2)
            power = float(power_str)
            
            # Format power nicely (avoid .0 for integers)
            if power == int(power):
                power_display = str(int(power))
            else:
                power_display = str(power)
            
            _resolve_external_variable(var_name, data)
            base_matrix = _handle_variable(var_name, data, drop_first)
            
            # Apply power transformation with validation
            transformed_matrix = {}
            for col_name, col_values in base_matrix.items():
                # Validate for negative powers (division by zero)
                if power < 0 and (col_values == 0).any():
                    invalid_indices = col_values[col_values == 0].index.tolist()
                    n_invalid = len(invalid_indices)
                    sample_indices = invalid_indices[:5]
                    raise ValueError(
                        f"Cannot compute {col_name}^{power}: found {n_invalid} zero value(s).\n"
                        f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                        f"Suggestions:\n"
                        f"  - Filter out zero values from your data\n"
                        f"  - Add a small constant: ({col_name} + c)^{power}\n"
                        f"  - Use a different transformation"
                    )
                # Validate for fractional powers (roots of negative numbers)
                elif 0 < power < 1 and (col_values < 0).any():
                    invalid_indices = col_values[col_values < 0].index.tolist()
                    n_invalid = len(invalid_indices)
                    sample_indices = invalid_indices[:5]
                    raise ValueError(
                        f"Cannot compute {col_name}^{power}: found {n_invalid} negative value(s).\n"
                        f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                        f"Suggestions:\n"
                        f"  - Filter out negative values from your data\n"
                        f"  - Take absolute value if appropriate\n"
                        f"  - Use a different transformation"
                    )
                
                transformed = col_values ** power
                transformed_matrix[f'{col_name}^{power_display}'] = transformed
            
            part_matrices.append(transformed_matrix)
        else:
            _resolve_external_variable(part, data)
            part_matrix = _evaluate_term(part, data, response_var, drop_first)
            part_matrices.append(part_matrix)
    
    # Compute interaction
    result = {}
    
    # Get all combinations of column names
    column_combinations = product(*[list(pm.keys()) for pm in part_matrices])
    
    for col_combo in column_combinations:
        # Create interaction column name
        interaction_name = ':'.join(col_combo)
        
        # Compute interaction values
        interaction_values = pd.Series(1.0, index=data.index)
        for i, col_name in enumerate(col_combo):
            interaction_values *= part_matrices[i][col_name]
        
        result[interaction_name] = interaction_values
    
    return result


def _apply_function(func_name, inner_term, data, response_var, drop_first):
    """Apply transformation function to term."""
    # Normalize 'inv' to 'inverse'
    if func_name == 'inv':
        func_name = 'inverse'
    
    inner_matrix = _evaluate_term(inner_term, data, response_var, drop_first)
    result = {}
    
    for col_name, col_values in inner_matrix.items():
        if func_name == 'log':
            if (col_values <= 0).any():
                invalid_indices = col_values[col_values <= 0].index.tolist()
                n_invalid = len(invalid_indices)
                sample_indices = invalid_indices[:5]
                raise ValueError(
                    f"Cannot compute log({col_name}): found {n_invalid} value(s) <= 0.\n"
                    f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                    f"Suggestions:\n"
                    f"  - Filter out non-positive values from your data\n"
                    f"  - Add a constant: log({col_name} + c)\n"
                    f"  - Use a different transformation"
                )
            transformed = np.log(col_values)
            result[f'log({col_name})'] = transformed
        elif func_name == 'sqrt':
            if (col_values < 0).any():
                invalid_indices = col_values[col_values < 0].index.tolist()
                n_invalid = len(invalid_indices)
                sample_indices = invalid_indices[:5]
                raise ValueError(
                    f"Cannot compute sqrt({col_name}): found {n_invalid} negative value(s).\n"
                    f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                    f"Suggestions:\n"
                    f"  - Filter out negative values from your data\n"
                    f"  - Take absolute value if appropriate\n"
                    f"  - Use a different transformation"
                )
            transformed = np.sqrt(col_values)
            result[f'sqrt({col_name})'] = transformed
        elif func_name == 'inverse':
            if (col_values == 0).any():
                invalid_indices = col_values[col_values == 0].index.tolist()
                n_invalid = len(invalid_indices)
                sample_indices = invalid_indices[:5]
                raise ValueError(
                    f"Cannot compute inverse({col_name}): found {n_invalid} zero value(s).\n"
                    f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                    f"Suggestions:\n"
                    f"  - Filter out zero values from your data\n"
                    f"  - Add a small constant: 1 / ({col_name} + c)\n"
                    f"  - Use a different transformation"
                )
            transformed = 1 / col_values
            result[f'inverse({col_name})'] = transformed
        elif func_name.startswith('pow__'):
            power = float(func_name.split('__')[1].replace('_', '.'))
            # Format power nicely (avoid .0 for integers)
            if power == int(power):
                power_display = str(int(power))
            else:
                power_display = str(power)
            transformed = col_values ** power
            result[f'{col_name}^{power_display}'] = transformed
        else:
            raise ValueError(f"Unknown function: {func_name}")
    
    return result


def _handle_variable(var_name, data, drop_first):
    """Handle a single variable, creating dummies if categorical."""
    _resolve_external_variable(var_name, data)
    
    col = data[var_name]
    
    if isinstance(col.dtype, CategoricalDtype) or col.dtype == object:
        # Create dummy variables and ensure they're float type
        dummies = pd.get_dummies(col, prefix=var_name, drop_first=drop_first)
        # Convert to float to avoid boolean masking issues
        return {col_name: dummies[col_name].astype(float) for col_name in dummies.columns}
    else:
        # Numerical variable
        return {var_name: col}


def _resolve_external_variable(var_name, data):
    """Resolve variable from external scope if not in data."""
    if var_name not in data.columns:
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the variable
            for _ in range(10):  # Limit depth to avoid infinite loops
                frame = frame.f_back
                if frame is None:
                    break
                
                if var_name in frame.f_locals:
                    data[var_name] = frame.f_locals[var_name]
                    return
                elif var_name in frame.f_globals:
                    data[var_name] = frame.f_globals[var_name]
                    return
            
            raise KeyError(f"Variable '{var_name}' not found in data or external scope.")
        finally:
            del frame


def _apply_response_transformation(response_var, resp_func, resp_power, data):
    """Apply transformation to response variable."""
    Y = data[response_var]
    
    if resp_func == 'power':
        # Handle power transformations
        if resp_power < 0:
            # Negative powers involve division
            if (Y == 0).any():
                invalid_indices = Y[Y == 0].index.tolist()
                n_invalid = len(invalid_indices)
                sample_indices = invalid_indices[:5]
                raise ValueError(
                    f"Cannot compute {response_var}^{resp_power}: found {n_invalid} zero value(s).\n"
                    f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                    f"Suggestions:\n"
                    f"  - Filter out zero values from your data\n"
                    f"  - Add a small constant: ({response_var} + c)^{resp_power}\n"
                    f"  - Use a different transformation"
                )
        elif 0 < resp_power < 1:
            # Fractional powers (roots) require non-negative values
            if (Y < 0).any():
                invalid_indices = Y[Y < 0].index.tolist()
                n_invalid = len(invalid_indices)
                sample_indices = invalid_indices[:5]
                raise ValueError(
                    f"Cannot compute {response_var}^{resp_power}: found {n_invalid} negative value(s).\n"
                    f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                    f"Suggestions:\n"
                    f"  - Filter out negative values from your data\n"
                    f"  - Take absolute value if appropriate\n"
                    f"  - Use a different transformation"
                )
        return Y ** resp_power
    elif resp_func == 'log':
        if (Y <= 0).any():
            invalid_indices = Y[Y <= 0].index.tolist()
            n_invalid = len(invalid_indices)
            sample_indices = invalid_indices[:5]
            raise ValueError(
                f"Cannot compute log({response_var}): found {n_invalid} value(s) <= 0.\n"
                f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                f"Suggestions:\n"
                f"  - Filter out non-positive values from your data\n"
                f"  - Add a constant: log({response_var} + c)\n"
                f"  - Use a different transformation"
            )
        return np.log(Y)
    elif resp_func == 'sqrt':
        if (Y < 0).any():
            invalid_indices = Y[Y < 0].index.tolist()
            n_invalid = len(invalid_indices)
            sample_indices = invalid_indices[:5]
            raise ValueError(
                f"Cannot compute sqrt({response_var}): found {n_invalid} negative value(s).\n"
                f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                f"Suggestions:\n"
                f"  - Filter out negative values from your data\n"
                f"  - Take absolute value if appropriate\n"
                f"  - Use a different transformation"
            )
        return np.sqrt(Y)
    elif resp_func == 'inverse':
        if (Y == 0).any():
            invalid_indices = Y[Y == 0].index.tolist()
            n_invalid = len(invalid_indices)
            sample_indices = invalid_indices[:5]
            raise ValueError(
                f"Cannot compute inverse({response_var}): found {n_invalid} zero value(s).\n"
                f"Sample indices: {sample_indices}{'...' if n_invalid > 5 else ''}\n"
                f"Suggestions:\n"
                f"  - Filter out zero values from your data\n"
                f"  - Add a small constant: 1 / ({response_var} + c)\n"
                f"  - Use a different transformation"
            )
        return 1 / Y
    else:
        return Y
