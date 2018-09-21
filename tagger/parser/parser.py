# coding: utf-8

import re
import nltk
import logging
import traceback
import pandas as pd
import unidecode
import lxml
import multiprocessing
from lxml.etree import ParserError
from lxml.html.clean import Cleaner


class Parser():

    def clean_html_corpus(self, html):
        """Clean html corpus.

        Args:
            html: content of a web page

        Returns:
            text: corpus of a web page cleaned
        """
        log = logging.getLogger('clean_html_corpus')

        try:
            if html is not None:
                regex = re.compile('<.*?>')
                cleaner = Cleaner(
                    style=True,
                    page_structure=True,
                    meta=False,
                    kill_tags=['footer', 'header', 'aside', 'menu', 'noscript'])
                # remove scripts, invalid tags, ...
                cleaned_html = cleaner.clean_html(html)
                # remove html tags
                regex = re.compile('<.*?>')
                text = re.sub(regex, '', cleaned_html)
                # remove accents
                text = unidecode.unidecode(text)
                # keep only words
                regex = re.compile('[a-zA-Z]{2,}')
                words = re.findall(regex, text)
                text = ' '.join(words)
                log.info("successfully cleaned html")

                return text
            else:
                return None
        except Exception as e:
            log.debug("type error: {}" .format(e))
            log.debug(traceback.format_exc())

    def extract_tags(self, raw_corpus, url, tag, domain):
        """Extract tags from html code.

        Args:
            raw_corpus: corpus of a web page

        Returns:
            url: url a of web page
            tag: tag of a web page
            domain: domain name of a web page
            lang: language of a web page
            corpus_marker: marker between words
            corpus: cleaned corpus of a web page
        """
        log = logging.getLogger('extract_tags')

        lang = keywords = description = title = []
        h = []

        try:
            doc = lxml.html.fromstring(str(raw_corpus))
            lang = doc.attrib.get('lang')
            h = ' bbb '.join(doc.xpath("//body//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::p]/text()"))
            title = ' '.join(doc.xpath("//head/title/text()"))
            description = ' '.join(doc.xpath('//head/meta[@name="description"]/@content'))
            keywords = ' '.join(doc.xpath('//head/meta[@name="keywords"]/@content'))
            corpus_marker = h+" bbb "+keywords+" bbb "+description+" bbb "+title
            corpus = self.clean_html_corpus(raw_corpus)
            log.info("successfully extracted tags of a html corpus")

            return url, tag, domain, lang, corpus_marker, corpus
        except ParserError as pe:
            # If the corpus is empty
            log.debug("type error: {}" .format(pe))
            log.debug(traceback.format_exc())

            return url, None, None, None, None
        except ValueError as ve:
            # If the document is in format xml instead of html
            log.debug("type error: {}" .format(ve))
            log.debug(traceback.format_exc())

            return url, None, None, None, None

    def find_language_url(self, url):
        """Get language from url.

        Args:
            url: url of a web page

        Returns:
            lang: language a of web page
        """
        log = logging.getLogger('find_language_url')

        url = url.replace('/', '.')
        url_split = url.split('.')
        url_split = set([i for i in url_split if len(i) == 2])

        try:
            # English
            if 'uk' in url_split or 'au' in url_split or 'ca' in url_split:
                return 'en'
            # Spanish
            elif 'ar' in url_split or 'mx' in url_split or 'es' in url_split or 'pe' in url_split or 've' in url_split or 'cl' in url_split:
                return 'es'
            # French
            elif 'fr' in url_split or 'be' in url_split:
                return 'fr'
            # Italian
            elif 'it' in url_split:
                return'it'
            else:
                return None
        except Exception as e:
            log.debug("type error: {}" .format(e))
            log.debug(traceback.format_exc())

    def aggregate_lang(self, meta_lang, lang_url):
        """Aggregation of language detected.

        Args:
            meta_lang: language of a web page in the html content
            lang_url: language of a web page in the url

        Returns:
            meta_lang: language of a web page
            lang_url: language of a web page
        """
        log = logging.getLogger('aggregate_lang')

        try:
            if lang_url is None:
                return meta_lang
            else:
                return lang_url
        except Exception as e:
            log.debug("type error: {}" .format(e))
            log.debug(traceback.format_exc())

    def detect_languages(self, df):
        """Takes as input dataframe with columns url and html and
        return a list of languages detected.

        Args:
            df: content of a web page

        Returns:
            lang: languages of a web page
        """
        log = logging.getLogger('detect_languages')

        try:
            # Get the language according to extension in domain name.
            lang_url = df['url'].apply(self.find_language_url)
            lang = list(map(lambda x, y: self.aggregate_lang(x, y), df['meta_lang'], lang_url))

            return lang
        except Exception as e:
            log.debug("type error: {}" .format(e))
            log.debug(traceback.format_exc())

    def get_words(self, raw_corpus):
        """Delete accents and other unicode issues, numbers and take words larger
        than 3.

        Args:
            raw_corpus: raw data

        Returns:
            corpus: cleaned corpus
        """
        log = logging.getLogger('get_words')

        try:
            if raw_corpus is None:
                return None
            else:
                corpus = unidecode.unidecode(raw_corpus)
                pattern = re.compile(r'(?!\d)[\w]{3,}')
                words = pattern.findall(corpus)

                return ' '.join([word.lower() for word in words if not any(c.isdigit() for c in word)])
        except Exception as e:
            log.debug("type error: {}" .format(e))
            log.debug(traceback.format_exc())

    def stem_corpus(self, corpus, lang):
        """Apply stemming on corpus.

        Args:
            corpus: corpus of a web page
            lang: language of a web page

        Returns:
            corpus: stemmed corpus
        """
        log = logging.getLogger('stem_corpus')

        try:
            corpus = corpus.replace("\\t", "")
            corpus = corpus.replace("\\n", "")
            corpus = corpus.replace("\\xa", "")
            corpus = re.sub("\d+", " ", corpus)
            pattern = re.compile(r'(?!\d)[\w\-]{3,}')
            corpus = pattern.findall(corpus)
            new_corpus = []
            if lang == 'fr':
                stemmer = nltk.SnowballStemmer("french")
                for word in corpus:
                    new_corpus.append(stemmer.stem(word))
            elif lang == 'en':
                stemmer = nltk.SnowballStemmer("english")
                for word in corpus:
                    new_corpus.append(stemmer.stem(word))
            elif lang == 'it':
                stemmer = nltk.SnowballStemmer("italian")
                for word in corpus:
                    new_corpus.append(stemmer.stem(word))
            elif lang == 'es':
                stemmer = nltk.SnowballStemmer("spanish")
                for word in corpus:
                    new_corpus.append(stemmer.stem(word))

            return ' '.join([item for item in new_corpus])
        except Exception as e:
            log.debug("type error: {}" .format(e))
            log.debug(traceback.format_exc())

    def delete_stop(self, sentence, stopword_list):
        """Delete stopwords in a sentence.

        Args:
            sentence: sentence words
            stopword_list: list of common stop words

        Returns:
            sentence: cleaned sentence
        """
        log = logging.getLogger('delete_stop')

        try:
            sentence = ' '.join([word for word in sentence.split() if word not in stopword_list])
        except AttributeError as ae:
            log.debug("Error while deleting stopwords: {}" .format(ae))
            log.debug(traceback.format_exc())
        return sentence

    def delete_stopen(self, sentence, stopword_list):
        """Delete stopwords in an english sentence.

        Args:
            sentence: sentence words
            stopword_list: list of common stop words

        Returns:
            sentence: cleaned sentence
        """
        log = logging.getLogger('delete_stopen')

        try:
            sentence = ' '.join([word for word in sentence.split() if word not in stopword_list])
        except AttributeError as ae:
            log.debug("Error while deleting english stopwords: {}" .format(ae))
            log.debug(traceback.format_exc())
        return sentence

    def delete_stopes(self, sentence, stopword_list):
        """Delete stopwords in a spanish sentence.

        Args:
            sentence: sentence words
            stopword_list: list of common stop words

        Returns:
            sentence: cleaned sentence
        """
        log = logging.getLogger('delete_stopes')

        try:
            sentence = ' '.join([word for word in sentence.split() if word not in stopword_list])
        except AttributeError as ae:
            log.debug("Error while deleting spanish stopwords: {}" .format(ae))
            log.debug(traceback.format_exc())
        return sentence

    def delete_stopit(self, sentence, stopword_list):
        """Delete stopwords in an italian sentence.

        Args:
            sentence: sentence words
            stopword_list: list of common stop words

        Returns:
            sentence: cleaned sentence
        """
        log = logging.getLogger('delete_stopit')

        try:
            sentence = ' '.join([word for word in sentence.split() if word not in stopword_list])
        except AttributeError as ae:
            log.debug("Error while deleting italian stopwords: {}" .format(ae))
            log.debug(traceback.format_exc())
        return sentence

    def drop_tags(self, list_tags):
        """Get useful tags.

        Args:
            listtags: list of tags of one url

        Returns:
            goodtags: tags without useless tags
        """
        log = logging.getLogger('drop_tags')

        goodtags = []

        try:
            for tag in list_tags:
                if int(tag) <= 1636:
                    goodtags.append(tag)
            return goodtags
        except Exception as e:
            log.debug("Error while cleaning tags: {}" .format(e))
            log.debug(traceback.format_exc())

    def extract_corpus(self, df, n_processes, stopwords, input_lang='fr', stemming=False):
        """Apply clean on raw data.

        Args:
            df: Takes as input a dataframe with 3 columns: html code and url and tags
            n_processes: number of processes
            stopwords: list of stopwords
            input_lang: language of a corpus
            stemming:

        Returns:
            df_final: Final DataFrame with corpus, lang & url
        """
        log = logging.getLogger('extract_corpus')

        log.info("Content of the not cleaned data frame: {}" .format(df[0]))

        pool = multiprocessing.Pool(processes=n_processes)
        df_final = pool.starmap(self.extract_tags, zip(
            df['text'], df['url'], df['tag'], df['domain']))
        pool.close()
        pool.join()

        df_final = pd.DataFrame(df_final)
        df_final.columns = ['url', 'tag', 'domain', 'meta_lang', 'text', 'text_maxime']

        df_final.fillna(value=pd.np.nan, inplace=True)
        df_final['lang'] = self.detect_languages(df_final)

        # Filter language
        df_final = df_final[df_final['lang'] == input_lang]

        df_final = df_final[pd.notnull(df_final['text_maxime'])]

        pool = multiprocessing.Pool(processes=n_processes)
        final_result = pool.map(self.get_words, df_final['text'])
        pool.close()
        pool.join()

        df_final['text'] = final_result
        df_final = df_final.drop(['meta_lang'], axis=1)
        df_final = df_final.dropna()

        stopword_list = stopwords[input_lang]
        df_final['text'] = df_final['text'].apply(self.delete_stop, args=(stopword_list,))
        # drop useless tags and specific to training stuff:
        df_final['tag'] = df_final['tag'].apply(self.drop_tags)
        df_final['corpus_x'] = df_final['text'].apply(lambda x: x.replace("bbb", "").strip())
        df_final['size'] = df_final['corpus_x'].apply(lambda x: len(x.split()))
        df_final = df_final[df_final['size'] >= 10]
        df_final.rename(columns={
            'corpus_x': 'corpus_clean',
            'text': 'corpus_marker'
            },
            inplace=True)

        log.info("Content of the cleaned data frame: {}" .format(df_final[0]))

        # in case we need to parse different languages
        #df_final_en = df_final.loc[df_final['lang'] == 'en']
        #df_final_en['corpus'] = df_final_en['corpus'].apply(delete_stopen)

        #df_final_es = df_final.loc[df_final['lang'] == 'es']
        #df_final_es['corpus'] = df_final_es['corpus'].apply(delete_stopes)

        #df_final_it = df_final.loc[df_final['lang'] == 'it']
        #df_final_it['corpus'] = df_final_it['corpus'].apply(delete_stopit)

        return df_final
