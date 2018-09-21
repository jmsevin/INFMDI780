# -*- coding: utf-8 -*-

"""
This is the basic Core configuration file.

All options are basic Python data types. You may use all of Python's language
capabilities to specify settings in this file.
"""

"""
[LAUNCHER]
Configuration parameters that control the scraping process. You will most
likely want to change these values.
"""
# How many processes to run a scraper jobs asynchronously
n_processes = 200

# compression type
compression_brotli = "BROTLI"

# Batch size
batch_size = 5000

# Path to stopwords data
path_stopwords = "../data/stopwords_fr.txt"

# aws access key
aws_access_key_id = ""  # insert your aws_access_key_id

# aws scret access key
aws_secret_access_key = ""  # insert your aws_secret_access_key

# bucket and key name
bucket = ""  # CONFIDENTIAL

# Path to upload files
path_to_upload_files = "" # CONFIDENTIAL