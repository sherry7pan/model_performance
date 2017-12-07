from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import model
import pickle



def refine_random_forest_classifier(X_train,y_train):
    '''Using grid search to tune the parameters of random forest classifier
       and save the model.
    INPUT:
        - X_train, y_train
    OUTPUT:
        - best estimator of random forest by grid search
    '''
    rf_grid = {
    'max_depth': [4, 8, None],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [ 2, 4],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'n_estimators': [50, 100, 200, 400],
    }

    rf_grid_cv = GridSearchCV(RandomForestClassifier(),rf_grid, scoring='roc_auc')
    rf_grid_cv.fit(X_train, y_train)

    best_params_rf=rf_grid_cv.best_params_
    best_model_rf= rf_grid_cv.best_estimator_

    filename = 'random_forest_classifier.pkl'
    pickle.dump(best_model_rf, open(filename, 'wb'))

    print best_params_rf
    return best_model_rf

def assess_performance(clf, X_test, y_test):
    '''
    Assess the performance of the predictive model
    INPUT
        - est: A predictive model
        - X (pandas dataframe or numpy array): The feature matrix
        - y (pandas series or numpy array): The target vector
    OUTPUT
        None
    '''

    print 'Model: {}'.format(clf.__class__.__name__)
    print '-' * 50

    # Calculate the out-of-sample AUC
    AUC_holdout = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    print 'The AUC score on the holdout data is {:.6f}'.format(AUC_holdout)

if __name__ == '__main__':
    print "*****************"*5
    print "This program takes over 10 minutes to run due to grid searching process."
    print "To interrupt the program, press CTRL+SHIFT+C "
    print "*****************"*5
    train_link = 'train.csv'
    # load data
    df = model.load_data(train_link)
    # process data
    df = model.data_processing(df)
    # undersample
    X,y = model.undersample(df,0.5)
    # train test split
    X_train, X_test, y_train, y_test = model.train_and_test_split(X,y)
    #refine classifier with grid search
    final_clf = refine_random_forest_classifier(X_train,y_train)
    # assess performance with holdout data
    assess_performance(final_clf, X_test, y_test)
