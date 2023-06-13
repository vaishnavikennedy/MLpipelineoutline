from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from logger_setup import setup_logger


# function to train and evaluate the given model using cross-validation
def train_evaluate_model(model, params, X_train_res, y_train_res, X_val, y_val):
    """
    Train and evaluate the given model using cross-validation
    """
    # Setup logger to print important information
    logger = setup_logger()

    # Perform grid search with cross-validation on the given model and parameters
    grid_search = GridSearchCV(model, params, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_res, y_train_res)

    # Print the best parameters and score found during grid search
    logger.info("Best parameters: %s ", grid_search.best_params_)
    logger.info("Best score: %s", grid_search.best_score_)

    # Make predictions on validation data and print evaluation metrics
    y_pred = grid_search.predict(X_val)
    logger.info("Validation accuracy: %s", accuracy_score(y_val, y_pred))
    logger.info("Confusion matrix: \n%s", confusion_matrix(y_val, y_pred))
    logger.info("Classification report: \n%s", classification_report(y_val, y_pred))

    # Return the grid search object for later use
    return grid_search
