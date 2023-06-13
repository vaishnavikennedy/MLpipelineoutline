import seaborn as sns
import matplotlib.pyplot as plt


# Define a function for finding correlated features using a correlation matrix
def corr_features(df, feature_importance):
    """
    Function for finding about the feature interaction  using a correlation matrix
    When correlated a pair of features  has to be removed,uses feature_importance to decide which one to remove
    based on its lower importance value.
    :param df: Dataframe
    :param feature_importance: feature_importance calculated using Random Forest
    :return: correlated_features: list of correlated features
    """
    # Compute the correlation matrix using the input dataframe
    corr_matrix = df.corr()

    # Plot the correlation matrix as a heatmap using seaborn
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()

    # Set a threshold for correlation coefficient
    threshold = 0.8
    correlated_features = []  # empty list to store the features

    # Loop over the upper triangle of the correlation matrix and check for correlated features
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]

                # Choose which feature to remove based on its feature importance score
                if feature_importance[col_i] > feature_importance[col_j]:
                    col_to_remove = col_j
                else:
                    col_to_remove = col_i

                # Add the feature to the list of correlated features if it hasn't already been added
                if col_to_remove not in correlated_features:
                    correlated_features.append(col_to_remove)

    # Return the list of correlated features for removal
    return correlated_features
