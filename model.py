import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import pickle

def load_data(link):
    '''
    load the data and skip rows that doesn't match
    the number of features in the dataset.

    INPUT:
        - a link to dataset
    OUTPUT:
        -  pandas dataframe
    '''
    df = pd.read_csv(link, error_bad_lines=False);
    return df

def data_processing(df):
    ''' Convert categorical columns, treat/drop missing values,

        INPUT:
            - original dataset (pandas dataframe)
        OUTPUT:
            - a processed dataset(pandas dataframe)
    '''

    #dropping the missing values in the gender column
    df= df[df['gender'].notnull()]
    #transform categorial columns into binary columns
    cols_to_transform = [ 'gender','device_type']
    df = pd.get_dummies( df,columns = cols_to_transform )
    #since categorial column gender only takes on two values, we drop one.
    df.drop('gender_M', axis=1, inplace=True)
    # adding a column 'vehicles_per_driver' to the existing dataframe.
    df['vehicles_per_driver'] = df['n_vehicles'] / df['n_drivers'].astype('float')
    return df

def undersample(df, target):
    """
    undersample function get X,y from dataframe and  randomly discards negative observations from
    X, y to achieve the target proportion
    INPUT:
    datafrmae- original dataset
    target - the intended percentage of positive
             class observations in the output
    OUTPUT:
    X_undersampled, y_undersampled

    `undersample` randomly discards negative observations from
    X, y to achieve the target proportion
    """

    #set outcome label array
    y= df.pop('outcome').values
    #set feature matrix
    X = df.values

    if target < sum(y)/float(len(y)):
        return X, y
    # determine how many negative (majority) observations to retain
    positive_count = sum(y)
    negative_count = len(y) - positive_count
    keep_count = positive_count*(1-target)/target
    keep_count = int(round(keep_count))
    # randomly discard negative (majority) class observations
    keep_array = np.array([True]*keep_count + [False]*(negative_count-keep_count))
    np.random.shuffle(keep_array)
    X_positive, y_positive = X[y==1], y[y==1]
    X_negative = X[y==0]
    X_negative_undersampled = X_negative[keep_array]
    y_negative = y[y==0]
    y_negative_undersampled = y_negative[keep_array]
    X_undersampled = np.vstack((X_negative_undersampled, X_positive))
    y_undersampled = np.concatenate((y_negative_undersampled, y_positive))

    return X_undersampled, y_undersampled
    print Counter(y_undersampled)


def train_and_test_split(X,y):
    ''' Create the final analytical dataset and perform
        train, test split

        INPUT:
            - dataset (pandas dataframe)
        OUTPUT:
            - train and test datasets
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=10, stratify = y)
    return X_train, X_test, y_train, y_test



def random_forest_classifier(X_train, y_train ):
    ''' Build a random_forest_regressor with cross validation.
        Print cross_validation score.
        INPUT:
            - X_train, y_train
        OUTPUT:
            - none
    '''
    #preparing folds for cross_validation
    fix_fold = StratifiedKFold(6, random_state=1).get_n_splits()
    # fit the model
    rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight='auto')
    print 'The cross-validated AUC for RandomForestClassifier is {:.6f} .'.format(np.mean(cross_val_score(rfc, X_train, y_train, cv= fix_fold, scoring='roc_auc')))


def prediction(clf, link_test_dataset):

    df_test = pd.read_csv(link_test_dataset)
    df_test = data_processing(df_test)
    X_test = df_test.values
    prediction = clf.predict_proba(X_test)[:,1]
    return prediction

if __name__ == '__main__':

    train_link = 'train.csv'
    # load data
    df = load_data(train_link)
    # process data
    df = data_processing(df)
    # undersample
    X,y = undersample(df,0.5)

    # train test split
    X_train, X_test, y_train, y_test = train_and_test_split(X,y)
    # build model and see initial accessment with cross_val_score
    random_forest_classifier(X_train, y_train )
    #refine classifier with grid search
