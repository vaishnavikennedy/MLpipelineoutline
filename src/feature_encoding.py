from sklearn.preprocessing import LabelEncoder
from logger_setup import setup_logger


# Define a function for label encoding categorical features
def label_encode(df):
    """
    Function for feature encoding (label encoding of categorical values)
    :param df: Dataframe
    :return: df: Label encoded Dataframe
    """
    # Set up a logger for logging messages during feature encoding
    logger = setup_logger()
    logger.info("Feature Encoding taking place \n")

    # Get the names of categorical columns in the input dataframe
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Loop over each categorical column and encode its values using LabelEncoder
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Return the label encoded dataframe
    return df
