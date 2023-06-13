# Function to check the consistency of the agebins column in the dataframe
def check_age_consistency(df):

    """
    Checks the consistency between the 'Age' column and the 'AgeBins' column in the input dataframe.
    :param df: pets df in which the consistency to be checked
    :return: None: prints error message if any inconsistency found
    """

    for index, row in df.iterrows():
        age = row['Age']
        age_bins = row['AgeBins']
        if age_bins == '[ 0, 2)':
            if age < 0 or age >= 2:
                print(f"Error: Age {age} does not fall within AgeBins {age_bins} at index {index}.")
        elif age_bins == '2':
            if age < 2 or age > 2:
                print(f"Error: Age {age} does not fall within AgeBins {age_bins} at index {index}.")
        elif age_bins == '[ 3, 6)':
            if age < 3 or age >= 6:
                print(f"Error: Age {age} does not fall within AgeBins {age_bins} at index {index}.")
        elif age_bins == '[ 6, 12)':
            if age < 6 or age >= 12:
                print(f"Error: Age {age} does not fall within AgeBins {age_bins} at index {index}.")
        elif age_bins == '[ 12, 24)':
            if age < 12 or age >= 24:
                print(f"Error: Age {age} does not fall within AgeBins {age_bins} at index {index}.")
        elif age_bins == '[ 24, 60)':
            if age < 24 or age >= 60:
                print(f"Error: Age {age} does not fall within AgeBins {age_bins} at index {index}.")
        elif age_bins == '[ 60,255]':
            if age < 60 or age > 255:
                print(f"Error: Age {age} does not fall within AgeBins {age_bins} at index {index}.")
