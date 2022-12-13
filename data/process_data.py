import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - messages data csv location
    categories_filepath - categories data csv location
    
    Provides a dataframe where the input datasets are merged on index
    '''
   
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets on index
    df = messages.merge(categories, left_index=True, right_index=True)
    
    return df
    


def clean_data(df):
    '''
    INPUT:
    df - dataframe of merged data to be cleaned
    
    Returns a dataframe where the each categorisation has its own column flagged 1 or 0 for that message
    '''
        
    categories = df.categories.str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # get the category names from the first row
    category_colnames = list(row.str[:-2])
    # change the dataframe column names to those of categories
    categories.columns = category_colnames
    
    # loop through each column and set it to its value for that category
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # merge our category columns on to our messages
    df = df[['message', 'genre']]
    df = pd.concat([df.reset_index(drop=True), categories.reset_index(drop=True)], axis=1)
    
    # remove related = 2 rows from dataset
    df = df[df.related < 2]
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''
    INPUT:
    df - dataframe to be put on database
    database_filename - the database location the data is to be stored
    
    Put the dataframe into a database at the location input
    '''
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False)


def main():
    
    '''
    Runs the module
    '''
        
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
