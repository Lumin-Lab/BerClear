import pickle
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score


def get_ber_class_mapping():
    """
    Returns a DataFrame that maps class names to BER classes.
    Returns:
        - class_names (list): A list of class names.
        - class_mappling (pandas.DataFrame): A DataFrame with two columns: 'rating' and 'class'.
            'rating' represents the class names and 'class' represents the BER classes.
    """
    # Define the fine bins for BER classes
    FINE_BINS = [
        'A1', 'A2', 'A3', 
        'B1', 'B2', 'B3', 
        'C1', 'C2', 'C3', 
        'D1', 'D2', 
        'E1', 'E2', 
        'F', 
        'G'
    ]
    
    # Enumerate the class names
    class_names = range(15)
    
    # Create a DataFrame to map class names to BER classes
    class_mapping = pd.DataFrame(
        zip(class_names, FINE_BINS),
        columns=['rating', 'class']
    )
    
    return class_names, class_mapping


def load_train_test(data_dir, data_type, data_format='csv'):
    """
    Load train and test data from the specified directory.

    Parameters:
    - data_dir (str): The directory path where the data files are located.
    - data_type (str): The type of data to load (e.g., 'train', 'test').
    - data_format (str, optional): The format of the data files (default: 'csv').

    Returns:
    - X_train (DataFrame): The training data without the target column.
    - y_train (Series): The target column of the training data.
    - X_test (DataFrame): The test data without the target column.
    - y_test (Series): The target column of the test data.
    """
    if data_format == 'parquet':
        df_train = pd.read_parquet(f'{data_dir}/{data_type}_train.parquet')
        df_test = pd.read_parquet(f'{data_dir}/{data_type}_test.parquet')
    if data_format == 'csv':
        df_train = pd.read_csv(f'{data_dir}/{data_type}_train.csv')
        df_test = pd.read_csv(f'{data_dir}/{data_type}_test.csv')      
    target_col = "EnergyRating"
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    return X_train, y_train, X_test, y_test


def load_embedding(data_dir):
    """
    Load the embedding data from the specified directory.
    Args:
        data_dir (str): The directory path where the embedding data is stored.
    Returns:
        tuple: A tuple containing the loaded training and testing embedding arrays.
    """
    train = np.load(f'{data_dir}/train.npy')
    test = np.load(f'{data_dir}/test.npy')   
    return train, test


def load_pickle(fpath):
    """
    Load a pickle file and return the loaded object.

    Parameters:
    fpath (str): The file path of the pickle file.

    Returns:
    object: The loaded object from the pickle file.
    """
    with open(fpath, 'rb') as file:
        obj = pickle.load(file)  
        return obj
    

def get_scores(y, predicted_y):
    """
    Calculate accuracy and F1 score for the predicted labels.

    Parameters:
    - y: The true labels.
    - predicted_y: The predicted labels.

    Returns:
    A dictionary containing the accuracy and F1 score.

    Example usage:
    >>> y = [0, 1, 0, 1]
    >>> predicted_y = [0, 1, 1, 1]
    >>> get_scores(y, predicted_y)
    {'acc': [0.75], 'f1': [0.6666666666666666]}
    """
    acc = accuracy_score(y, predicted_y)
    # display(acc) #for notebook
    f1 = f1_score(y, predicted_y, average='macro')
    # display(f1) #for notebook
    return {
        'acc': [acc],
        'f1': [f1]
    }


def save_model_pickle(output_dir, clf, data_type, clf_model):
    """
    Save the trained classifier model as a pickle file.

    Parameters:
    - output_dir (str): The directory where the pickle file will be saved.
    - clf (object): The trained classifier object to be saved.
    - data_type (str): The type of data used to train the classifier.
    - clf_model (str): The name of the classifier model.

    Returns:
    - None
    """
#     os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/{clf_model}_{data_type}.pkl', 'wb') as file:
        pickle.dump(clf, file)
        
    

