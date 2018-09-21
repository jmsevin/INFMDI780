# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time
import json
import logging
import fnmatch
import threading
import requests
import traceback
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer


# PARAMETERS
path = 'data'
TOPIC_URL = 'url'
TOPIC_HTML = 'html'
KAFKA_HOST = 'localhost:9092'
KAFKA_OFFSET = 'earliest'
KAFKA_CONSUMER_TIMEOUT = 100


def get_files(input_path):
    """Get all parquet's files from a repository.

    Args:
        input_path: path to repository

    Returns:
        files: list of all files
    """
    log = logging.getLogger('get_files')

    files = []
    start_time = time.time()
    try:
        if os.path.isfile(input_path) and input_path.endswith(".parquet"):
            files.append(input_path)
        elif os.path.isdir(input_path):
            for file in os.listdir(input_path):
                if fnmatch.fnmatch(file, '*.parquet'):
                    files.append(input_path+file)
        log.debug("=>   get files usage time: {}s".format(time.time() - start_time))
        return files
    except Exception as ex:
        log.debug("type error: {}" .format(ex))
        log.debug(traceback.format_exc())


def scraper(url):
    """Scrape a specific url.

    Args:
        url: link to a web site

    Returns:
        text: html content of an url
    """
    log = logging.getLogger('scraper')

    start_time = time.time()
    try:
        response = requests.get(url)
        if response.status_code in [200, 201]:
            text = response.text
            log.debug("=> scrape usage time: {}s".format(time.time() - start_time))
            response.connection.close()
            return text
    except requests.ConnectionError as ce:
            log.debug("Connection Error. Make sure you are connected to Internet: {}" .format(ce))
            log.debug(traceback.format_exc())
    except requests.Timeout as to:
            log.debug("Timeout Error: {}" .format(to))
            log.debug(traceback.format_exc())
    except requests.RequestException as re:
            log.debug("General Error: {}" .format(re))
            log.debug(traceback.format_exc())
    except KeyboardInterrupt:
            log.debug("Someone closed the program")
    except Exception as e:
            log.debug("type error: {}" .format(e))
            log.debug(traceback.format_exc())
    return None


class Producer(threading.Thread):
    daemon = True

    def run(self):
        log = logging.getLogger('run_producer')

        producer = KafkaProducer(
            bootstrap_servers=KAFKA_HOST,
            value_serializer=lambda v: json.dumps(v).encode())

        print('Loading function host:{} topic:{}', producer, TOPIC_URL)

        files = (get_files(path))
        start_time = time.time()
        try:
            for filename in files:
                data = pd.read_parquet(filename)
                for row in data.itertuples():
                    contents = row.site_page
                    producer.send(TOPIC_URL, contents)
            producer.flush()
            producer.close(timeout=10)
            print("=>   send usage time: {}s".format(time.time() - start_time))
        except Exception as ex:
            log.debug("type error: {}" .format(ex))
            log.debug(traceback.format_exc())
            raise ex


class Consumer(threading.Thread):
    daemon = True

    def run(self):
        log = logging.getLogger('run_consumer')

        producer = KafkaProducer(
            bootstrap_servers=KAFKA_HOST,
            key_serializer=str.encode,
            value_serializer=lambda v: json.dumps(v).encode())

        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_HOST,
            auto_offset_reset=KAFKA_OFFSET,
            consumer_timeout_ms=KAFKA_CONSUMER_TIMEOUT,
            value_deserializer=lambda m: json.loads(m.decode()))
        consumer.subscribe(TOPIC_URL)

        start_time = time.time()
        try:
            for message in consumer:
                log.debug("consume url: {} in topic: {}" .format(message.value, TOPIC_URL))
                corpus = scraper(message.value)
                producer.send(TOPIC_HTML, key=message.value, value=corpus)
                producer.flush()
                log.debug("=>   send html: {} in topic: {}" .format(corpus, TOPIC_HTML))
            log.debug("=>   consumer URL usage time: {}s".format(time.time() - start_time))
            consumer.close()
        except Exception as ex:
            log.debug("type error: {}" .format(ex))
            log.debug(traceback.format_exc())
            raise ex


def main():
    threads = [
        Producer(),
        Consumer()
    ]

    for t in threads:
        t.start()

    time.sleep(10)

    # wait for threads to finish
    for t in threads:
        t.join()


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s.%(msecs)s:%(name)s:%(thread)d:' +
               '%(levelname)s:%(process)d:%(message)s',
        level=logging.INFO
    )
    main()
