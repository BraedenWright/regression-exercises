import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    # First as int to get rid of '.0'
    df.year_built = df.year_built.astype(int)
    
    # Then as object for categorical sorting
    df.year_built = df.year_built.astype(object)
    df.fips = df.fips.astype(object)

    # Download cleaned data to a .csv
    df.to_csv(filename, index=False)

    return df


def split_data(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=1313)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=1313)

    # Take a look at your split datasets

    print(f'train <> {train.shape}')
    print(f'validate <> {validate.shape}')
    print(f'test <> {test.shape}')
    return train, validate, test



def scale_data(train, validate, test, scaler, return_scaler=False):
    '''
    This function takes in train, validate, and test dataframes and returns a scaled copy of each.
    If return_scaler=True, the scaler object will be returned as well
    '''
    
    num_columns = ['bedrooms', 'bathrooms', 'sqr_feet', 'tax_value', 'taxamount']
    
    train_scaled = train.copy()
    validated_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler.fit(train[num_columns])
    
    train_scaled[num_columns] = scaler.transform(train[num_columns])
    validate_scaled[num_columns] = scaler.transform(validate[num_columns])
    test_scaled[num_columns] = scaler.transform(test[num_columns])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

    
    
    
    
def wrangle_grades():
    """
    (Given by Madeleine Capper)
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    """
    # Acquire data from csv file.
    grades = pd.read_csv("student_grades.csv")
    # Replace white space values with NaN values.
    grades = grades.replace(r"^\s*$", np.nan, regex=True)
    # Drop all rows with NaN values.
    df = grades.dropna()
    # Convert all columns to int64 data types.
    df = df.astype("int")
    return df




def get_telco_data():

    filename = 'telco.csv'
    
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)

    database = 'telco_churn'
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
     
    query = '''
            SELECT * 
            FROM customers
            JOIN contract_types USING(contract_type_id)
            JOIN internet_service_types USING(internet_service_type_id)
            JOIN payment_types USING(payment_type_id)
            '''
     
    df = pd.read_sql(query, url)
    df.to_csv(filename, index=False)


    print('Pulling from SQL...')
    return df



def prep_telco(df):
    '''
    Takes in the telco df and cleans the dataframe up for use. Also changes total_charges from type(obj) to type(float) and removes any accounts with a tenure of 0 to keep information relevant
    '''
    
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    df.total_charges = df.total_charges.replace(' ', 0).astype(float)
    df = df[df.tenure != 0]

    cat_columns = ['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'contract_type', 'internet_service_type', 'payment_type']

    for col in cat_columns:
        telco_dummy = pd.get_dummies(df[col],
                                     prefix=df[col].name,
                                     dummy_na=False,
                                     drop_first = True)
        df = pd.concat([df, telco_dummy], axis=1)
        df = df.drop(columns=[col])
    
    return df