import numpy as np
import pandas as pd
from ravix.modeling.format_utils import format_sigfigs, format_r_style, format_pvalue

def print_anova_table(model, typ=None):
    """
    Prints the ANOVA table for a given model.
    Args:
        model: A fitted statsmodels regression model.
        typ: If 1, returns Type I (sequential) ANOVA.
             If None, returns overall ANOVA table.
    """
    
    if typ == 1:
        # Manual calculation of Type I (sequential) ANOVA
        from scipy import stats
        
        y = model.model.endog
        X = model.model.exog
        n = len(y)
        y_mean = np.mean(y)
        
        # Get variable names from model
        param_names = model.params.index.tolist()
        
        # Calculate sequential sum of squares
        results = []
        var_names = []
        
        # Track which column is the intercept
        intercept_idx = None
        for i, name in enumerate(param_names):
            if name.lower() in ['intercept', 'const']:
                intercept_idx = i
                break
        
        # Start with intercept-only model if there is one
        for i in range(len(param_names)):
            # Skip the intercept in output
            if i == intercept_idx:
                continue
            
            # Fit model with variables up to position i
            X_subset = X[:, :(i+1)]
            
            # Calculate fitted values
            params_subset = np.linalg.lstsq(X_subset, y, rcond=None)[0]
            y_pred = X_subset @ params_subset
            
            # Calculate sum of squares for this model
            ss_model = np.sum((y_pred - y_mean) ** 2)
            
            # If this is not the first variable after intercept, calculate incremental SS
            if i > 0:
                X_prev = X[:, :i]
                params_prev = np.linalg.lstsq(X_prev, y, rcond=None)[0]
                y_pred_prev = X_prev @ params_prev
                ss_prev = np.sum((y_pred_prev - y_mean) ** 2)
                ss_incremental = ss_model - ss_prev
            else:
                # First non-intercept variable
                if intercept_idx is not None:
                    # There's an intercept, so calculate from intercept-only model
                    X_intercept = X[:, [intercept_idx]]
                    params_intercept = np.linalg.lstsq(X_intercept, y, rcond=None)[0]
                    y_pred_intercept = X_intercept @ params_intercept
                    ss_intercept = np.sum((y_pred_intercept - y_mean) ** 2)
                    ss_incremental = ss_model - ss_intercept
                else:
                    ss_incremental = ss_model
            
            # Calculate df (1 for each variable)
            df = 1
            
            # Calculate mean square
            ms = ss_incremental / df
            
            # Calculate F-statistic
            mse_resid = model.mse_resid
            f_stat = ms / mse_resid
            
            # Calculate p-value
            p_value = 1 - stats.f.cdf(f_stat, df, model.df_resid)
            
            var_names.append(param_names[i])
            results.append([df, ss_incremental, ms, f_stat, p_value])
        
        # Add residual row
        var_names.append('Residual')
        results.append([int(model.df_resid), np.sum(model.resid ** 2), model.mse_resid, '', ''])
        
        # Create DataFrame
        anova_df = pd.DataFrame(results, 
                                columns=['df', 'Sum Sq', 'Mean Sq', 'F', 'Pr(>F)'],
                                index=var_names)
        
        # Apply formatting to numeric columns
        for col in ['Sum Sq', 'Mean Sq', 'F']:
            anova_df[col] = anova_df[col].apply(format_r_style)
        
        # Special formatting for p-values
        anova_df['Pr(>F)'] = anova_df['Pr(>F)'].apply(format_pvalue)
        
        return anova_df
    
    # Original overall ANOVA table
    # Number of observations
    n_obs = int(model.nobs)
    # F-statistic and its p-value
    f_statistic = model.fvalue
    f_p_value = model.f_pvalue
    # Degrees of freedom for the model and residuals
    df_model = int(model.df_model)
    df_resid = int(model.df_resid)
    df_total = df_model + df_resid
    # R-squared and Adjusted R-squared
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    # Root Mean Squared Error (Root MSE)
    root_mse = np.sqrt(model.mse_resid)
    # Sum of Squares
    ssr = np.sum(model.resid ** 2)  # Residual Sum of Squares
    ssm = np.sum((model.fittedvalues - np.mean(model.model.endog)) ** 2)  # Model Sum of Squares
    sst = ssr + ssm  # Total Sum of Squares
    # Mean Squares
    msr = ssr / df_resid  # Residual Mean Square
    msm = ssm / df_model  # Model Mean Square
    
    # Create ANOVA table
    anova_table = pd.DataFrame({
        'df': [df_model, df_resid, df_total],
        'Sum Sq': [ssm, ssr, sst],
        'Mean Sq': [msm, msr, ''],
        'F': [f_statistic, '', ''],
        'Pr(>F)': [f_p_value, '', '']
    }, index=['Regression', 'Residual', 'Total'])
    
    # Apply formatting to numeric columns
    for col in ['Sum Sq', 'Mean Sq', 'F']:
        anova_table[col] = anova_table[col].apply(format_r_style)
    
    # Special formatting for p-values
    anova_table['Pr(>F)'] = anova_table['Pr(>F)'].apply(format_pvalue)
    
    return anova_table
