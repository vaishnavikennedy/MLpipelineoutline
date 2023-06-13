from _csv import Error
import requests
import pandas as pd
from io import BytesIO
import zipfile
from logger_setup import setup_logger


# Function to read csv files from the zipped folder in the url
def ingest_data(url_link, filename):
    """
    Function to read csv files from the zipped folder in the url
    param url_link: Url from where the data sets need to be fetched(dropbox)
    param filename: name of the csv file to be fetched
    param dataframe: dataframe created for the csv file
    return: dataframe: dataframe created for the csv file
    """
    logger = setup_logger()
    try:
        logger.info(" Data Ingestion In Progress. Data ingested from given URL")
        r = requests.get(url_link)
        files = zipfile.ZipFile(BytesIO(r.content))
        dataframe = pd.read_csv(files.open(filename))
    except Error:
        logger.error(" Data Ingestion Failed due to CSV error")

    return dataframe
