import os
import pickle

from data_types import PostData

PROCESSED_DATA_PATH = './processed_data'


def load_data() -> list[PostData]:
    posts = []
    for filename in os.listdir(PROCESSED_DATA_PATH):
        with open(f'{PROCESSED_DATA_PATH}/{filename}', 'rb') as file:
            post = pickle.load(file)
            posts.append(post)
    return posts
