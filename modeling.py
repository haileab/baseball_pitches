import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def split(df):
    """
    Test/Train split and standardized normalization of the features.
    """
    train, test = train_test_split(df, random_state=0, test_size=0.30, shuffle=False)
    y_train = train.pop('target')
    X_train = train
    y_test = test.pop('target')
    X_test = test
    # getting geatures label for later
    features_labels = X_train.columns

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_std,y_train, X_test_std, y_test, features_labels

def clf(X_std,y_train, X_test_std, y_test):
    """
    Random forest classification model.
    """
    classifier = RandomForestClassifier(n_estimators = 100,  random_state = 0, n_jobs= -1)
    model = classifier.fit(X_std, y_train)
    y_pred = model.predict(X_test_std)

    return X_std, y_test, y_pred, model

def print_results(X_std, y_test, y_pred, features_labels, model):
    """
    Prints the accuracy, precision, recall, and f1-score for the classification model.
    """
    print("-" * 10, 'Random Forest' )
    print("Accuracy = ",accuracy_score(y_test,y_pred))
    print("\n")
    print(classification_report(y_test, y_pred))

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_std.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                features_labels[indices[f]],
                                importances[indices[f]]))

if __name__ == "__main__":
    import data_prep
    df = data_prep.main()
    X_std,y_train, X_test_std, y_test, features_labels = split(df)
    X_std, y_test, y_pred, model = clf(X_std,y_train, X_test_std, y_test)
    print_results(X_std,y_test, y_pred, features_labels, model)
