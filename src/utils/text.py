import re
import warnings
from typing import List
from functools import lru_cache
import funcy as fp
from .. import settings


SPLIT_PATTERN = re.compile(r"(\s|[-/,;.()])")

IDENTIFY_HTML_TAGS = re.compile(r'<[^>]*>', re.IGNORECASE)
IDENTIFY_URLS = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+))', re.IGNORECASE)
IDENTIFY_REPEATED_SPACES = re.compile(r'\s{2, }', re.IGNORECASE)
IDENTIFY_HASH_TAGS = re.compile(r'(^|\s)([#][\w_-]+)', re.IGNORECASE)
IDENTIFY_MENTIONS = re.compile(r'(^|\s)([@][\w_-]+)', re.IGNORECASE)
IDENTIFY_NUMBERS = re.compile(r'([$]?[0-9]+,*[0-9]*)+', re.IGNORECASE)

IDENTIFY_REPETITIONS = r'(.)\1+'
MODIFY_REPETITIONS = r'\1\1'


def clean_text(text: str, unify_html_tags: bool = True, unify_urls: bool = True, trim_repeating_spaces: bool = True,
               unify_hashtags: bool = True, unify_mentions: bool = True, unify_numbers: bool = True,
               trim_repeating_letters: bool = True, lower_case: bool = False, remove_stopwords: bool = False):

    text = re.sub(IDENTIFY_HTML_TAGS, ' ', text) if unify_html_tags else text
    text = re.sub(IDENTIFY_URLS, 'URL', text) if unify_urls else text
    text = re.sub(IDENTIFY_REPEATED_SPACES, ' ', text) if trim_repeating_spaces else text
    text = re.sub(IDENTIFY_HASH_TAGS, ' HASH_TAG ', text) if unify_hashtags else text
    text = re.sub(IDENTIFY_MENTIONS, ' USERNAME ', text) if unify_mentions else text
    text = re.sub(IDENTIFY_NUMBERS, ' NUMBER ', text) if unify_numbers else text
    text = re.sub(IDENTIFY_REPETITIONS, MODIFY_REPETITIONS, text) if trim_repeating_letters else text
    text = text.lower() if lower_case else text
    if remove_stopwords:
        text = " ".join(fp.remove(set(load_stopwords()), re.split(SPLIT_PATTERN, text)))

    return text


@lru_cache()
def load_stopwords() -> List[str]:

    try:
        with open(settings.STOPWORDS_PATH, 'r') as file:
            return file.read().split('\n')
    except (OSError, IOError) as e:
        warnings.warn('Stopwords were not loaded.')
        return []
