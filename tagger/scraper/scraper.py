# -*- coding: utf-8 -*-

import logging
import traceback
import requests
import pandas as pd
import multiprocessing


class Scraper():

    def scrape_url(self, url, tag, domain):
        """Scrape multiple web pages.

        Args:
            url: url of a web page
            tag: label of a web page
            domain: domain name of a web page

        Returns:
            dict: python dictionary of urls, tags and domains
        """
        log = logging.getLogger('scrape_url')

        try:
            response = requests.get(url, timeout=1)
            if response.status_code in [200, 201]:
                return {
                    'url': url,
                    'text': response.text,
                    'tag': tag,
                    'domain': domain
                    }
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
        return {
            'url': url,
            'text': None,
            'tag': tag,
            'domain': domain
            }

    def run_job(self, df, n_processes):
        """Get scrape job by url.

        Args:
            df: url of a web page
            n_processes: number of processes

        Returns:
            df: url, tag and domain
        """
        log = logging.getLogger('run_job')

        results = []
        urls, tags, domains = self.read_data(df)

        pool = multiprocessing.Pool(processes=200)
        for url, tag, domain in zip(urls, tags, domains):
            r = pool.apply_async(self.scrape_url, args=(url, tag, domain))
            results.append(r)
        output = [i.get() for i in results]
        pool.close()
        pool.join()

        df = pd.DataFrame(output)
        log.info("successfully scraped dataframe: {}" .format(df))

        return df
