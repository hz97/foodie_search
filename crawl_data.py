import os
import pickle
from typing import Any

from instagrapi import Client
from instagrapi.types import Media

from timing import timing

ACCOUNT_USERNAME = os.environ.get('IG_USERNAME')
ACCOUNT_PASSWORD = os.environ.get('IG_PASSWORD')
RAW_DATA_PATH = './raw_data'


def read_usernames() -> list[str]:
    with open('usernames.txt') as file:
        usernames = [line.strip() for line in file]
    return usernames


def get_raw_post_data(media: Media) -> dict[str, Any]:
    return {'id': media.code,
            'text': media.caption_text,
            'taken_at': media.taken_at,
            'comment_count': media.comment_count or 0,
            'like_count': media.like_count or 0}


@timing
def crawl_data() -> None:
    client = Client()
    client.login(ACCOUNT_USERNAME, ACCOUNT_PASSWORD)
    usernames = read_usernames()
    for username in usernames:
        user_id = client.user_id_from_username(username)
        medias = client.user_medias(user_id, amount=50)
        for media in medias:
            with open(f'{RAW_DATA_PATH}/{media.code}', 'wb') as file:
                raw_post_data = get_raw_post_data(media)
                pickle.dump(raw_post_data, file)
