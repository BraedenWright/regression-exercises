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



    
    
    
def rfe_feature_rankings(x_scaled, x, y, k):
    '''
    Takes in the predictors, the target, and the number of features to select,
    and it should return a database of the features ranked by importance
    '''
    
    # Make it
    lm = sklearn.linear_model.LinearRegression()
    rfe = sklearn.feature_selection.RFE(lm, n_features_to_select=k)

    # Fit it
    rfe.fit(x_scaled, y)
    
    var_ranks = rfe.ranking_
    var_names = x_train.columns.tolist()
    ranks = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    ranks = ranks.sort_values(by="Rank", ascending=True)
    return ranks




def rfe(x_scaled, x, y, k):
    '''
    Takes in the predictors, the target, and the number of features to select,
    and it should return the top k features based on the RFE class. 
    '''
    # Make it
    lm = sklearn.linear_model.LinearRegression()
    rfe = sklearn.feature_selection.RFE(lm, n_features_to_select=k)

    # Fit it
    rfe.fit(x_scaled, y)
    
    # Use it
    features_to_use = x.columns[rfe.support_].tolist()
    
    #
    #all_rankings = show_features_rankings(x, rfe)
    
    return features_to_use




def select_kbest(x_scaled, x, y, k):
    '''
    Takes in an x and y dataframe and the number of features to select
    and returns the names of the top features based on SelectKBest
    '''
    # Make it
    kbest = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_regression, k=k)

    # Fit tit
    kbest.fit(x_scaled, y)
    
    # Use it 
    return x.columns[kbest.get_support()].tolist()