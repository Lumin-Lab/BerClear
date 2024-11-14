from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

def get_clf_mlp():
    """
    Returns a multi-layer perceptron (MLP) classifier model and its type.

    Returns:
        clf (MLPClassifier): The MLP classifier model.
        clf_model (str): The name of the classifier model.
    """
    # Rest of the code...
    # mlp_dropout = 0.01
    mlp_epochs= 30
    mlp_batch_size = 256
    mlp_learning_rate = 1.0e-03
    mlp_hidden_layers = [256, 128, 64, 32]

    clf_model = 'mlp'
    clf = MLPClassifier(
        random_state=42, 
        hidden_layer_sizes=mlp_hidden_layers, 
    #     activation='relu', 
    #     solver='adam', 
        batch_size=mlp_batch_size, 
        learning_rate='constant', 
        learning_rate_init=mlp_learning_rate, 
        max_iter=mlp_epochs
    )
    return clf, clf_model


def get_clf_rf():
    """
    Returns a random forest classifier and its model name.

    Returns:
        clf (RandomForestClassifier): A random forest classifier object.
        clf_model (str): The model name of the classifier.
    """
    clf_model = 'randomforest'
    clf = RandomForestClassifier(max_depth=10, random_state=0)
    return clf, clf_model




def get_pred_probs_sklearn(X_train, y_train, nrows=5000, n_splits=5):
    """
    Calculate the predicted probabilities using a sklearn classifier.

    Parameters:
    - X_train (array-like): The input training data.
    - y_train (array-like): The target training data.
    - nrows (int, optional): The number of rows to consider from the training data. Defaults to 5000.
    - n_splits (int, optional): The number of folds in cross-validation. Defaults to 5.

    Returns:
    - pred_probs (array-like): The predicted probabilities for each class.
    """
    clf, clf_model = get_clf_mlp()
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    if nrows is None: nrows = X_train.shape[0]
    pred_probs = cross_val_predict(
            clf,
            X_train[:nrows, :],
            y_train[:nrows],
            cv=kf,
            method="predict_proba",
        )
    return pred_probs


def get_pred_probs(model, X_test):
    """
    Calculate the predicted probabilities for each class label using the given model.

    Parameters:
        model (object): The trained model used for prediction.
        X_test (array-like): The input data for which the predicted probabilities are calculated.

    Returns:
        array-like: The predicted probabilities for each class label.
    """
    pred_probs = model.predict_proba(X_test)
    return pred_probs