from logger_setup import setup_logger


# Function for outlier analysis
def outlier_categorical(df):
    """
    # Function for finding the rare categories
    :param df: Dataframe
    :return: rare_categories less frequent categories
    """
    logger = setup_logger()

    logger.info("Since most features are categorical like and through EDA we know no need of outlier analysis ,"
                "outliers in categorical values done by "
                "finding rare categories \n")
    # Loop through each categorical col for the outlier analysis
    for col in df.select_dtypes(include=['object']):
        freq = df[col].value_counts(normalize=True)  # Calculate the frequency of each category
        threshold = 0.01
        rare_categories = freq[freq < threshold].index.tolist()  # rare categories less frequently than threshold
        if len(rare_categories) > 0:
            logger.info(f"Rare categories in {col}: {', '.join(rare_categories)}")

    return rare_categories
