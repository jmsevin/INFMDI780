# -*- coding: utf-8 -*-

import time
import click
import s3fs
import pandas as pd
import unidecode
import logging
import traceback
import multiprocessing
import pyarrow as pa
import pyarrow.parquet as pq
from random import shuffle
import math
from scraper.scraper import Scraper
from parser.parser import Parser
import config
import subprocess

#########################################################################
#                                                                       #
#                                                                       #
# To run: python core.py <MONTH> <START_DAY> <END_DAY> <URLS_PER_DAYS>  #
#                                                                       #
#########################################################################


def get_data_aws(self, filesname):
    """Get fil.

    Args:
        filesname: files

    Returns:
        data: dataframe
    """
    log = logging.getLogger('get_data_aws')

    data = pd.DataFrame()
    for file in filesname:
        data = data.append(pd.read_parquet('s3://' + config.bucket + file))
    data.rename(columns={"tag": "google_segments", "domain": "site_domain"}, inplace=True)
    data.drop_duplicates('url', inplace=True)
    log.debug("dataframe: {}" .format(data))

    return data


def get_stopwords(filename):
    """Get stopwords.

    Args:
        filename: filename with stopwords

    Returns:
        stopwords: list of stopwords
    """
    log = logging.getLogger('get_stopword')

    stopwords = pd.read_csv(filename, header=None)
    stopwords = [unidecode.unidecode(word) for word in stopwords[0]]
    stopwords = set(stopwords)
    stopwords = {'fr': stopwords}
    log.debug("stopwords: {}" .format(stopwords))

    return stopwords


def read_data(self, data):
    """Read data from a dataframe.

    Args:
        data: dataframe

    Returns:
        urls: list of urls
        tags: list of tags
        domains: list of domains
    """
    log = logging.getLogger('read_data')

    try:
        urls = data[:, 0]
        tags = data[:, 1]
        domains = data[:, 2]
        log.info("successfully read data")
        log.debug("content urls: {}" .format(urls.head(1)))
        log.debug("content tags: {}" .format(tags.head(1)))
        log.debug("content domains: {}" .format(domains.head(1)))
    except Exception as e:
        log.debug("type error: {}" .format(e))
        log.debug(traceback.format_exc())
    return urls, tags, domains


def spider(df, n_processes, stopwords):
    """Launch scraper and parser.

    Args:
        df: url stored in a data frame
        n_processes: process to run a launcher

    Returns:
        table: corpus of a web page
    """
    log = logging.getLogger('spider')

    scraper = Scraper()
    parser = Parser()
    df = scraper.scrape_job(df, n_processes)
    log.debug("dataframe scraped: {}" .format(df.head(1)))

    df_final = parser.extract_corpus(df, 2*multiprocessing.cpu_count(), stopwords)
    table = pa.Table.from_pandas(df_final)
    log.debug("dataframe parsed: {}" .format(table))

    return table


def launch_job(month, start_day, end_day, urls_per_day, s3, my_bucket, random=False):
    """Launch scraper and parser.

    Args:
        month: month to start scraping
        start_day: starting point
        end_day: ending point
        urls_per_day: number of url to scrape by jobs
        s3: s3
        my_bucket: bucket name
        random: shuffle files

    """
    log = logging.getLogger('launch_job')

    stopwords = get_stopwords(config.path_stopwords)

    for day in range(start_day, end_day + 1):
        day = "%02d" % (day,)

        input_path = my_bucket
        input_path = input_path + 'm=' + month + '/d=' + day
        files_name = s3.ls(input_path)

        if random:
            shuffle(files_name)

        j = 100
        total_length = 0
        i = 2
        while total_length < urls_per_day:
            if(files_name[i] != config.bucket + 'm=' + month + '/d=' + day + '/_SUCCESS'):
                data = pd.read_parquet("s3://"+files_name[i])
                data = data.sample(frac=1)
                size = data.shape[0]
                epoch = math.ceil(size/config.batch_size)
                for k in range(0, size, config.batch_size):
                    if(total_length < urls_per_day):
                        url_start = k
                        url_end = k+config.batch_size
                        jj = month + '_' + day + '_' + str(j)
                        df = data[['site_page', 'google_segments', 'site_domain']].values[int(url_start):int(url_end)]

                        table = spider(df, config.n_processes, stopwords)

                        filename_to_upload = 'data_{}.parquet'.format(jj)
                        pq.write_table(table, filename_to_upload, compression=config.compression_brotli)
                        s3.put(filename_to_upload, config["path_to_upload_files"] + filename_to_upload)
                        subprocess.check_output(["rm", str(filename_to_upload)])

                        j += 1
                        total_length += config.batch_size
                    else:
                        break
                url_start = epoch*config.batch_size
                i += 1
            else:
                i += 1


@click.command()
@click.argument('month', envvar="MONTH")
@click.argument('start_day', envvar="START_DAY")
@click.argument('end_day', envvar="END_DAY")
@click.argument('urls_per_day', envvar="URLS_PER_DAY")
def main(month, start_day, end_day, urls_per_day):
    # Configure logging to show the name of the function
    # where the log message originates.
    logging.basicConfig(
        format='%(asctime)s.%(msecs)s:%(name)s:%(thread)d:' +
        '%(levelname)s:%(process)d:%(message)s',
        level=logging.INFO
    )

    month = "%02d" % (int(month),)
    start_day = int(start_day)
    end_day = int(end_day)
    urls_per_day = int(urls_per_day)

    s3 = s3fs.S3FileSystem(key=config.aws_access_key_id, secret=config.aws_secret_access_key)
    launch_job(month, start_day, end_day, urls_per_day, s3, config.bucket)


if __name__ == '__main__':
    main()
