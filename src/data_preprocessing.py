from sklearn.ensemble import RandomForestClassifier
from check_age_consistency import check_age_consistency
from check_fee_consistency import check_fee_consistency
from logger_setup import setup_logger
from outlier_analysis import outlier_categorical
from feature_encoding import label_encode
from feature_interaction import corr_features
from feature_selection import selected_features


# To carry out the Data Preprocessing for the Data Frame

def data_preprocess(df, drop_columns):
    """
    To Carry out the Data Preprocessing required for given pets data frame

    param: df: dataframe ingested
    return: df : the processed dataframe

    """
    logger = setup_logger()

    logger.info(" \n Data Preprocessing in Progress --------------- \n%s")

    # Missing Value Analysis
    missing_values_count = df.isnull().sum()
    logger.info('The missing values in df \n%s', missing_values_count)

    # Check Consistencies
    check_age_consistency(df)
    check_fee_consistency(df)

    # Feature engineering - Adding new feature
    logger.info("Adding new feature called RescuerPopularity \n")
    rescuer_popularity = df['RescuerID'].value_counts(normalize=True)
    df['RescuerPopularity'] = df['RescuerID'].map(rescuer_popularity)

    # Dropping unwanted/redundant columns
    logger.info("Dropping unwanted/redundant features \n%s")
    df = df.drop(drop_columns, axis=1)

    # Duplicates
    duplicates_count = df.duplicated().sum()
    logger.info('The duplicates values in df, dropping keeping first  \n %s', duplicates_count)
    df = df.drop_duplicates(keep='first')

    # Outlier Analysis
    outlier_categorical(df)

    # Features Encoding
    df = label_encode(df)

    # feature scaling and transformation
    df.describe()
    df.var()
    logger.info("Since the range of values not very varying and the variance is also in reasonable range, we dont need "
                "feature scaling or feature transformation \n%s ")

    # Calculate feature importance using RandomForestClassifier
    logger.info("Calculating feature importance using RandomForestClassifier \n%s")
    x = df.drop(columns=['AdoptionSpeed'])
    y = df['AdoptionSpeed']
    rf = RandomForestClassifier()
    rf.fit(x, y)
    feature_importance = {feature: importance for feature, importance in zip(x.columns, rf.feature_importances_)}
    logger.info("Feature importance: \n%s", feature_importance)

    # Feature Interaction - correlation matrix
    correlated_features = corr_features(df, feature_importance)  # Pass feature_importance to the function
    logger.info("The correlated features are \n%s", correlated_features)

    # Feature Selection - using SelectKBest
    best_features = selected_features(df)
    logger.info(" the selected features for training are \n %s", best_features)

    return df
