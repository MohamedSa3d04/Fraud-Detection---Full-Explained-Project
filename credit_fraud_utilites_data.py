import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
def load_data(file_path, target_column):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    return X.values, y.values

def split_data(X, y, test_size=0.2, random_state=40):
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    # Initialize the RobustScaler
    scaler = RobustScaler()
    
    # Fit the scaler on the training data and transform both training and testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def preprocess_data(file_path, target_column, test_size = 0.2):
    # Frist loading the data
    X, y = load_data(file_path, target_column)

    # Secondly splitting the data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    # Finally scaling the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Return the preprocessed data
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_shapes(X_train_scaled, X_test_scaled, y_train, y_test):
    # Print the shapes of the preprocessed data
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

def get_classes_dist (target):
    # Print the number of values of each class
    positive_numbers = sum(target)
    negative_numbers = len(target) - positive_numbers
    print(f'P : {positive_numbers} - N : {negative_numbers}')
    return positive_numbers, negative_numbers

def UnderSampling_data(X, y, factor = 1):
    # Under sampling the Majoirty class (Majority = n(Minority) * factor)
    minority_number, _ = get_classes_dist(y)
    rus = RandomUnderSampler(sampling_strategy={1:minority_number, 0:minority_number * factor})
    sampled_X, sampled_y = rus.fit_resample(X, y)
    print("New Data Dist. ")
    get_classes_dist(sampled_y)
    return sampled_X, sampled_y

def OverSampling_data(X, y, factor = 0.1):
    # Over sampling the Minority class (Minority = n(Majority) * factor)
    _, majority_number = get_classes_dist(y)
    ros = SMOTE(sampling_strategy={1:int(majority_number * factor), 0:majority_number})
    sampled_X, sampled_y = ros.fit_resample(X, y)
    print("New Data Dist. ")
    get_classes_dist(sampled_y)
    return sampled_X, sampled_y

