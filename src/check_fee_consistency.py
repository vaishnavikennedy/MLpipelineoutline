def check_fee_consistency(df):
    """
    Checks the consistency between the 'Fee' column and the 'FeeBins' column in the input dataframe.
    :param df: pets df in which the consistency to be checked
    :return: None: prints error message if any inconsistency found
    """
    for index, row in df.iterrows():
        fee = row['Fee']
        fee_bins = row['FeeBins']
        if fee_bins == '0':
            if fee != 0:
                print(f"Error: Fee {fee} does not fall within FeeBins {fee_bins} at index {index}.")
        elif fee_bins == '[ 1, 100)':
            if fee < 1 or fee >= 100:
                print(f"Error: Fee {fee} does not fall within FeeBins {fee_bins} at index {index}.")
        elif fee_bins == '[ 100, 200)':
            if fee < 100 or fee >= 200:
                print(f"Error: Fee {fee} does not fall within FeeBins {fee_bins} at index {index}.")
        elif fee_bins == '[ 200,3000]':
            if fee < 200 or fee > 3000:
                print(f"Error: Fee {fee} does not fall within FeeBins {fee_bins} at index {index}.")
