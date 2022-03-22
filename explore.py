import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Functions I've used to explore data



def plot_categorical_and_continuous_vars(cat_var, cont_var, df):
    '''
    This function accepts your dataframe and the name of the columns that hold 
    the continuous and categorical features and outputs 3 different plots 
    for visualizing a categorical variable and a continuous variable.
    '''
    
    # Title
    plt.suptitle(f'{cont_var} by {cat_var}')
                 
    # Lineplot
    sns.lineplot(x=cat_var, y=cont_var, data=df)
    plt.xlabel(cat_var)
    plt.ylabel(cont_var)
    
    # Swarm Plot
    sns.catplot(x=cat_var, y=cont_var, data=df, kind='swarm', palette='Greens')
    plt.xlabel(cat_var)
    plt.ylabel(cont_var)
    
    # Box Plot
    sns.catplot(x=cat_var, y=cont_var, data=df, kind='box', palette='Blues')
    plt.xlabel(cat_var)
    plt.ylabel(cont_var)
    
    # Bar Plot
    sns.catplot(x=cat_var, y=cont_var, data=df, kind='bar', palette='Purples')
    plt.xlabel(cat_var)
    plt.ylabel(cont_var)
    
    # Scatter plot with regression line
    sns.lmplot(x=cat_var, y=cont_var, data=df)
    plt.xlabel(cat_var)
    plt.ylabel(cont_var)
    
    plt.show()



# pairplot

def plot_variable_pairs(train, columns, hue=None):
    '''
    The function takes in a df and a list of columns from the df
    and displays a pair plot wid a regression line.
    '''
    
    kws = {'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}}
    sns.pairplot(train[columns],  kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})
    plt.show()
    
    

    
# *Work in progress*


