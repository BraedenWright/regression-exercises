import pandas as pd
import numpy as np



# One function to get SSE, MSE, and RMSE for both baseline and model

def regression_metrics(residual, baseline_residual, df):
    '''
    Function takes in the residuals from a regression model, the baseline regression, and the dataframe they are coming from,
    and produces an SSE, MSE, and RMSE for the model and baseline, print the results for easy comparison.
    '''
    
    # Get R^2 first
    #---------------------------
    # Model
    df['residual^2'] = df.residual**2
    # Baseline
    df['baseline_residual^2'] = df.baseline_residual**2
    
    
    # Square of Sum Errors (SSE)
    #----------------------------
    # Model
    SSE = df['residual^2'].sum()
    # Baseline
    Baseline_SSE = df['baseline_residual^2'].sum()

    
    # Mean Square Errors (MSE)
    #----------------------------
    # Model
    MSE = SSE/len(df)
    # Baseline
    Baseline_MSE = Baseline_SSE/len(df)
    
    
    # Root Mean Squared Error (RMSE)
    #-----------------------------
    # Model
    RMSE = sqrt(MSE)
    # Baseline
    Baseline_RMSE = sqrt(Baseline_MSE)

    print(f'SSE')
    print(f'-----------------------')
    print(f'Model SSE --> {SSE:.1f}')
    print(f'Baseline SSE --> {Baseline_SSE:.1f}')
    print(f'MSE')
    print(f'-----------------------')
    print(f'Model MSE --> {MSE:.1f}')
    print(f'Baseline MSE --> {Baseline_MSE:.1f}')
    print(f'RMSE')
    print(f'-----------------------')
    print(f'Model RMSE --> {RMSE:.1f}')
    print(f'Baseline RMSE --> {Baseline_RMSE:.1f}')


