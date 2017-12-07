import pickle
import pandas as pd
import model

def load_trained_model(filename):
    '''
    load the trained classifier from pickle file.
    INPUT
        - pickle file

    OUTPUT
        - unpickled trained classifer
    '''
    classifier = pickle.load(open( filename, "rb" ))
    return classifier

def prediction(clf, link_test_dataset):
    '''
    make prediction on the actual test set and return a dataframe with
    with features and predicted values in probability form.
    INPUT
        - clf: A predictive model
        - X (pandas dataframe or numpy array): The feature matrix

    OUTPUT
        - csv file with predicted valus
    '''

    df_test = pd.read_csv(link_test_dataset)
    df_test= model.data_processing(df_test)
    X_test = df_test.values
    df_test['predicted_values'] = clf.predict_proba(X_test)[:,1]

    return df_test.to_csv('prediction.csv', sep=',')


if __name__ == '__main__':
    random_forest_classifier = load_trained_model('random_forest_classifier.pkl')
    prediction(random_forest_classifier,'test.csv')
