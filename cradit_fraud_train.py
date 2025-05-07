#internal modules for data and model helping
from credit_fraud_utilites_data import *
from credit_fraud_utilites_eval import *

#models used
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

from joblib import dump # For saving the model as .joblib


#
def train_model(X, y, model_type='logistic'):
    '''
    X: Features Inputs
    y: Target
    model_type: Type of the model (defualt: 'logistic', 'neural_network', 'random_forest', 'XGBClassifier')
    '''

    if model_type == 'logistic':
        model = LogisticRegression(random_state=40)
    elif model_type == 'neural_network':
        model = MLPClassifier()
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=62, max_depth=13)
    elif model_type == 'XGBClassifier':
        positive_numbers, negative_numbers = get_classes_dist(y)
        scale_percentage = negative_numbers / positive_numbers
        model = XGBClassifier(n_estimators=50, max_depth=7, eval_metric=f1_score, scale_pos_weight = scale_percentage, reg_lambda=15)
    else:
        raise Exception("Un Valid Model!")


    model.fit(X, y)
    #Predict values (Training)
    predicted = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    #Training Evaluation
    print(f"{model_type} Metrices For Train: ")
    basic_metrics(y, predicted)
    get_best_thershold(predicted, probas)
    return model, predicted, probas

def train_voting_classifier(X, y, models_list):
    vclsf = VotingClassifier(estimators=models_list, voting='soft', n_jobs=2)
    vclsf.fit(X, y)

    return vclsf


if __name__ == '__main__':
    model_dict = {
        'threshold' : None, # Best thereshold value 
        'model' : None,
        'scaler' : None,
        'model_name' : 'logistic_model_for_fraud'
    }
    # Define the path and target column
    file_path = './data/fraud.csv'
    target_column = 'Class'

    # Preprocess the data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(file_path, target_column)
    
    # Print the shapes of the preprocessed data
    get_shapes(X_train_scaled, X_test_scaled, y_train, y_test)

    # Train Logistic Model
    logistic_model, predicted, probas = train_model(X_train_scaled, y_train, 'logistic')
    logistic_best_thershold = evaluate_model(logistic_model, X_test_scaled, y_test)
   
    # Save best logistic model
    model_dict['model'] = logistic_model
    model_dict['scaler'] = scaler
    model_dict['threshold'] = logistic_best_thershold
    model_dict['model_name'] = 'Logistic Model'
    # dump(model_dict, './models/logist_model_dict.joblib')

    
    # Let's now train a model in 50-50 data (Under Sampling)
    unsampled_X, unsampled_y = UnderSampling_data(X_train_scaled, y_train, 15)
    us_model, us_predicted, us_probas = train_model(unsampled_X, unsampled_y, 'logistic')
    us_best_threshold = evaluate_model(us_model, X_test_scaled, y_test, 0.7)

    model_dict['model'] = us_model
    model_dict['scaler'] = scaler
    model_dict['threshold'] = us_best_threshold
    model_dict['model_name'] = 'Under Sampling data + Logistic Model'
    # dump(model_dict, './models/us_model_dict.joblib')

    # Let's now train a model based on over-sampled data
    ovsampled_X, ovsampled_y = OverSampling_data(X_train_scaled, y_train, 1)
    os_model, os_predicted, os_probas = train_model(ovsampled_X, ovsampled_y)
    os_best_thershold = evaluate_model(os_model, X_test_scaled, y_test)


    model_dict['model'] = os_model
    model_dict['scaler'] = scaler
    model_dict['threshold'] = os_best_thershold
    model_dict['model_name'] = 'Over Sampling data + Logistic Model'
    # dump(model_dict, './models/os_model_dict.joblib')

    # Let's train neural network
    nn_model, nn_predicted, nn_probas = train_model(X_train_scaled, y_train, 'neural_network')
    nn_best_threshold = evaluate_model(nn_model, X_test_scaled, y_test)


    model_dict['model'] = nn_model
    model_dict['scaler'] = scaler
    model_dict['threshold'] = nn_best_threshold
    model_dict['model_name'] = 'Neural Network Model'
    # dump(model_dict, './models/nn_model_dict.joblib')

    
    # Let's train RandomForest model
    rf_model, rf_predicted, rf_probas = train_model(X_train_scaled, y_train, 'random_forest')
    rf_best_threshold = evaluate_model(rf_model, X_test_scaled, y_test)

    model_dict['model'] = rf_model
    model_dict['scaler'] = scaler
    model_dict['threshold'] = rf_best_threshold
    model_dict['model_name'] = 'Random Forest Model'
    # dump(model_dict, './models/rf_model_dict.joblib')

    # Let's train XGBoosting (XGClassifier) model
    xgb_classifier, xgc_predicted, xgc_probas = train_model(X_train_scaled, y_train, 'XGBClassifier')
    xgb_best_threshold = evaluate_model(xgb_classifier, X_test_scaled, y_test)
   
    model_dict['model'] = xgb_classifier
    model_dict['scaler'] = scaler
    model_dict['threshold'] = xgb_best_threshold
    model_dict['model_name'] = 'XGB Classifier Model'
    # dump(model_dict, './models/xgb_model_dict.joblib')
    
    # Votine Techinque (Soft Type)
    list_estimators = [
    ('logistic', logistic_model),
    ('neural_network', nn_model),
    ('random_forest', rf_model),
    ('xgb_classifier', xgb_classifier)
    ]

    vclf_model = train_voting_classifier(X_train_scaled, y_train, list_estimators)
    vclf_best_threshold = evaluate_model(vclf_model, X_test_scaled, y_test)

    model_dict['model'] = vclf_model
    model_dict['scaler'] = scaler
    model_dict['threshold'] = vclf_best_threshold
    model_dict['model_name'] = 'Voting Classifier'
    dump(model_dict, './models/vlcf_model_dict.joblib')
    
