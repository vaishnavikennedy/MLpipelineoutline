import numpy as np
import matplotlib.pyplot as plt


# Plots the feature importance in a bar chart.
def plot_feature_importance(importance, feature_names, X):
    """
    Plots the feature importance in a bar chart.

    :param X:The input features DataFrame
    :param importance: A list of feature importance
    :param feature_names: A list of feature names
    """
    # Sort the importance in descending order
    indices = np.argsort(importance)[::-1]

    # Create the bar chart
    plt.figure(figsize=(15, 5))
    plt.title("Feature importance")
    plt.bar(range(X.shape[1]), importance[indices],
            color="r", align="center")

    # Set x-axis labels and rotate them for better readability
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)

    # Set the x-axis limits
    plt.xlim([-1, X.shape[1]])

    # Show the plot
    plt.show()
