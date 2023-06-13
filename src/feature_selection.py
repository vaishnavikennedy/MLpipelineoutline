from sklearn.feature_selection import SelectKBest, chi2


# Define a function for feature selection using SelectKBest
def selected_features(df):
    """
    # Function for Feature Selection using SelectKBest model
    :param df: Dataframe
    :return: selected_features for training purpose
    """
    # Set the number of features to select and split the data into input (X) and output (y)
    k = 20
    x = df.drop(columns=['AdoptionSpeed'])
    y = df['AdoptionSpeed']

    # Initialize a SelectKBest object using the chi-squared test as the scoring function
    selector = SelectKBest(chi2, k=k)

    # Fit the selector to the input and output data, and get the names of the selected features
    selector.fit(x, y)
    select_features = x.columns[selector.get_support()]

    # Return the names of the selected features for training purposes
    return select_features
