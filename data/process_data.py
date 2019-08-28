import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Args:
        messages_filepath: Filepath to the messages dataset
        categories_filepath: Filepath to the categories dataset
    Returns:
        Pandas DataFrame with messages and categories merged on 'id'
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    raw_data_df = messages.merge(categories, on='id')
    return raw_data_df


def clean_data(df):
    """
    Args:
        df: Pandas DataFrame
    Returns:
        Cleaned Pandas DataFrame
    """
    categories = df['categories'].str.split(';',expand=True)
    category_row = categories[:1]
    category_colnames = category_row.apply(lambda x: x.str.split('-')[0][0], axis=0)
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1] if int(x.split('-')[1]) < 2 else 1)
        categories[column] = categories[column].astype(int)
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories], axis=1)
    df = df.drop_duplicates(keep='first')
    return df


def save_data(df, database_filename):
    """
    Args:
        df: Pandas DataFrame
        database_filename: sqlite database filename 
            to store the Pandas DataFrame
    Returns:
        Nothing
    """
    engine = create_engine('sqlite:///'+ database_filename) 
    df.to_sql('cleanData', engine, index=False)

def main():
    print(sys.argv)
    print(len(sys.argv))
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