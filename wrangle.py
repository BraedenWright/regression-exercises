import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from env import user, password, host
import warnings
warnings.filterwarnings('ignore')


def wrangle_zillow():
    ''' 
    This function pulls data from the zillow database from SQL and cleans the data up by changing the column names and romoving rows with null values.  
    Also changes 'fips' and 'year_built' columns into object data types instead of floats, since they are more catergorical, after which the dataframe is saved to a .csv.
    If this has already been done, the function will just pull from the zillow.csv
    '''

    filename = 'zillow.csv'
    
    if os.path.exists(filename):
        print('Reading cleaned data from csv file...')
        return pd.read_csv(filename)

    url = f"mysql+pymysql://{user}:{password}@{host}/zillow"

    query = '''
        SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
        FROM properties_2017
        LEFT JOIN propertylandusetype USING(propertylandusetypeid)
        WHERE propertylandusedesc IN ('Single Family Residential', 'Inferred Single Family Residential')
        '''

    df = pd.read_sql(query, url)

    # Rename columns
    df = df.rename(columns = {
                          'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'sqr_feet',
                          'taxvaluedollarcnt':'tax_value', 
                          'yearbuilt':'year_built'})
    
    # Drop null values
    df = df.dropna()

    # Change the dtype of 'year_built' and 'fips'
    df.year_built = df.year_built.astype(object)
    df.fips = df.fips.astype(object)

    # Download cleaned data to a .csv
    df.to_csv(filename, index=False)

    return df



