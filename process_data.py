import os
import pickle
import re
from collections import Counter
from typing import Any

import Stemmer
from cleantext import clean
from nltk import word_tokenize
from nltk.corpus import stopwords

from data_types import PostData
from timing import timing

RAW_DATA_PATH = './raw_data'
PROCESSED_DATA_PATH = './processed_data'

# 100 most common words in English + NLTK stop words
COMMON_WORDS = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
                'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
                'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we',
                'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all',
                'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
                'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make',
                'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
                'person', 'into', 'year', 'your', 'good', 'some', 'could',
                'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only',
                'come', 'its', 'over', 'think', 'also', 'back', 'after',
                'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
                'even', 'new', 'want', 'because', 'any', 'these', 'give',
                'day', 'most', 'us'} | set(stopwords.words('english'))
STEMMER = Stemmer.Stemmer('english')


def to_lowercase(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def filter_common_words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in COMMON_WORDS]


def clean_up(tokens: list[str]) -> list[str]:
    tokens = [clean(token, no_line_breaks=True, no_urls=True, no_emails=True,
                    no_phone_numbers=True, no_numbers=True, no_digits=True,
                    no_currency_symbols=True, no_punct=True, no_emoji=True,
                    replace_with_punct='', replace_with_url='', replace_with_email='',
                    replace_with_phone_number='', replace_with_number='',
                    replace_with_digit='', replace_with_currency_symbol='', lang='en') for token in tokens]
    tokens = [re.sub(r'@\w+', '', token) for token in tokens]  # remove mentions
    return [token for token in tokens if token]


def stem_words(tokens: list[str]) -> list[str]:
    return STEMMER.stemWords(tokens)


def process(text: str) -> list[str]:
    text = to_lowercase(text)
    tokens = tokenize(text)
    tokens = filter_common_words(tokens)
    tokens = clean_up(tokens)
    tokens = stem_words(tokens)
    return tokens


def compute_term_frequencies(tokens: list[str]) -> dict[str, int]:
    n = len(tokens)
    term_frequencies = Counter(tokens)
    term_frequencies = [tf / n for tf in term_frequencies]
    return term_frequencies


def process_raw_data(raw_post_data: dict[str, Any]) -> PostData:
    tokens = process(raw_post_data['text'])
    term_frequencies = compute_term_frequencies(tokens)
    return PostData(id=raw_post_data['id'],
                    term_frequencies=term_frequencies,
                    processed_text=' '.join(tokens),
                    taken_at=raw_post_data['taken_at'],
                    comment_count=raw_post_data['comment_count'],
                    like_count=raw_post_data['like_count'])


@timing
def process_data() -> None:
    for filename in os.listdir(RAW_DATA_PATH):
        with open(f'{RAW_DATA_PATH}/{filename}', 'rb') as file:
            raw_post_data = pickle.load(file)
            post = process_raw_data(raw_post_data)
        with open(f'{PROCESSED_DATA_PATH}/{filename}', 'wb') as file:
            pickle.dump(post, file)
