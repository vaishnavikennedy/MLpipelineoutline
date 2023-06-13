from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
from load_config import load_config
from data_ingestion import ingest_data
from data_preprocessing import data_preprocess
from feature_selection import selected_features
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from logger_setup import setup_logger
from train_evaluate_model import train_evaluate_model
import numpy as np
from feature_importances import plot_feature_importance

# LoggerSettings
logger = setup_logger()

# Loading the configuration file required for ML
config = load_config('config.yml')
logger.info("the loaded configurations are:", config)

# Data  Ingestion
pets_prepared_df = ingest_data(config["url"], "pets_prepared.csv")
breed_labels_df = ingest_data(config["url"], "breed_labels.csv")
state_labels_df = ingest_data(config["url"], "state_labels.csv")
color_labels_df = ingest_data(config["url"], "color_labels.csv")

# Data Preprocessing
try:
    pets_prepared_df = data_preprocess(pets_prepared_df, config["drop_fields"])
    good_features = selected_features(pets_prepared_df)
    X = pets_prepared_df.drop(columns=['AdoptionSpeed'])
    y = pets_prepared_df['AdoptionSpeed']
except KeyError as e:
    logger.error(f"Could not find column: {e}")
    raise e
except Exception as e:
    logger.error(f"Error occurred during data preprocessing: {e}")
    raise e

# Train Test Validate Split of the given dataframe.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_ratio"],
                                                    random_state=config["randomstate"])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=config["validate_ratio"] / config["train_ratio"],
                                                  random_state=config["randomstate"])

# Data Imbalance - the target value class distribution balanced using SMOTE
try:
    smote = SMOTE(random_state=config["randomstate"])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)  # Resample training data using SMOTE
    balanced_dist = pd.Series(y_train_resampled).value_counts()  # Check the resampled class distribution
    logger.info("Balanced class distribution:\n%s", balanced_dist)
except ValueError as e:
    logger.error(f"Error occurred during SMOTE resampling: {e}")
    raise e
except Exception as e:
    logger.error(f"Unexpected error occurred during SMOTE resampling: {e}")
    raise e

# Define models for the ML pipeline
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'Neural Network': MLPClassifier(max_iter=1000, early_stopping=True),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier()
}

# Define parameters for the Models
params = {
    'Decision Tree': {'max_depth': config['max_depth_dt']},
    'Random Forest': {'n_estimators': config['n_estimators_rf'], 'max_depth': config['max_depth_rf']},
    'Gradient Boosting': {'learning_rate': config['learning_rate_gb'], 'n_estimators': config['n_estimators_gb'],
                          'max_depth': config['max_depth_gb']},
    'SVM': {'C': config['c_svm'], 'gamma': config['gamma_svm'], 'kernel': config['kernel_svm']},
    'Neural Network': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh'], 'solver': ['adam'],
                       'learning_rate': ['constant']},
    'XGBoost': {'learning_rate': config['learning_rate_xgboost'], 'max_depth': config['max_depth_xgboost'],
                'n_estimators': config['n_estimators_xgboost']},
    'LightGBM': {'learning_rate': config['learning_rate_lightgbm'], 'max_depth': config['max_depth_lightgbm'],
                 'n_estimators': config['n_estimators_lightgbm']},
    'CatBoost': {'learning_rate': config['learning_rate_catboost'], 'max_depth': config['max_depth_catboost']}

}

# Train and evaluate each model using cross-validation and select the best one
best_models = {}
for name, model in models.items():  # iterate each model
    try:
        logger.info("Training model:%s", name)
        best_models[name] = train_evaluate_model(model, params[name], X_train_resampled, y_train_resampled, X_val,
                                                 y_val)  # pass the required parameters to train_evaluate_model
    except Exception as e:
        logger.error(f"Error occurred while training model {name}: {e}")
        raise e

if not best_models:
    logger.error("No suitable models found")
    raise ValueError("No suitable models found")

best_model = best_models[max(best_models, key=lambda key: best_models[key].best_score_)]
logger.info("Best model:%s ", best_model)  # best model chosen based on best score achieved during evaluation

# Evaluate the best model on the test set and print out the results
y_pred = best_model.predict(X_test)
logger.info("Test accuracy:%s ", accuracy_score(y_test, y_pred))
logger.info("Confusion matrix: \n%s", confusion_matrix(y_test, y_pred))
logger.info("Classification report: \n%s", classification_report(y_test, y_pred))

# Analyze the feature importance of the best-performing model
best_estimator = best_model.best_estimator_  # Store the best estimator from the best model in a variable
if hasattr(best_estimator, "feature_importances_"):
    feature_importances = best_estimator.feature_importances_  # if the there is feature_importance assign
else:
    result = permutation_importance(best_estimator, X_test, y_test, n_repeats=10, random_state=config["randomstate"])
    feature_importances = result.importances_mean  # if the there is no feature_importance use permutation_importance

plot_feature_importance(feature_importances, X.columns.tolist(), X)

# Adopter's preferences
# Make recommendations for improving pet listings
sorted_indices = np.argsort(feature_importances)[::-1]  # sort the feature importances in descending order
top_10_features = [X.columns[i] for i in sorted_indices[:10]]  # select the top 10 features based on their importance
logger.info("Top 10 features influencing adoption rate: %s", top_10_features)

# Save the best model with its respective best parameters
best_estimator = best_model.best_estimator_
model_name = type(best_estimator).__name__
filename = model_name + '_Bestofmodel.sav'
joblib.dump(best_model, filename)
loaded_model = joblib.load(filename)  # this is to load the model and check if the save is working
y_pred_sav = loaded_model.predict(X_test)
logger.info(" Yes , the saved model is working and loading Job \n %s", y_pred_sav)

# save all others models
for name, model in models.items():
    best_estimator = model
    model_name = type(best_estimator).__name__
    filename = model_name + '.sav'
    joblib.dump(model, filename)

# Ensemble Model
# define the models to ensemble
rf = RandomForestClassifier(n_estimators=best_models['Random Forest'].best_params_['n_estimators'],
                            max_depth=best_models['Random Forest'].best_params_['max_depth'])
gb = GradientBoostingClassifier(learning_rate=best_models['Gradient Boosting'].best_params_['learning_rate'],
                                n_estimators=best_models['Gradient Boosting'].best_params_['n_estimators'],
                                max_depth=best_models['Gradient Boosting'].best_params_['max_depth'])
dt = DecisionTreeClassifier(max_depth=best_models['Decision Tree'].best_params_['max_depth'])
svm = SVC(C=best_models['SVM'].best_params_['C'],
          gamma=best_models['SVM'].best_params_['gamma'],
          kernel=best_models['SVM'].best_params_['kernel'])
voting_clf = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('dt', dt), ('svm', svm)], voting="hard")
voting_clf.fit(X_train_resampled, y_train_resampled)  # Fit the ensemble model on the resampled training data
y_pred_ensemble = voting_clf.predict(X_test)  # Predict on the test set using the ensemble model
logger.info("Ensemble model accuracy:%s", accuracy_score(y_test, y_pred_ensemble))  # Evaluate the ensemble model
logger.info("Ensemble model confusion matrix:\n%s", confusion_matrix(y_test, y_pred_ensemble))
logger.info("Ensemble model classification report:\n%s", classification_report(y_test, y_pred_ensemble))
filename = 'ensemble_model.sav'  # Save the model
joblib.dump(voting_clf, filename)
