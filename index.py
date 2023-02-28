import math
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer

from data_types import PostData, SearchType, RankType
from process_data import process
from timing import timing


class Index:

    def __init__(self, posts: list[PostData]):
        self.index: dict[str, set[str]] = {}
        self.id_to_post: dict[str, PostData] = {}
        self.posts = posts
        for post in posts:
            self._index_post(post)

        self.vectorizer = TfidfVectorizer()
        tf_idf = self.vectorizer.fit_transform([post.processed_text for post in posts])
        self.tf_idf = pd.DataFrame(tf_idf.T.toarray(),
                                   index=self.vectorizer.get_feature_names_out())

    def _index_post(self, post: PostData) -> None:
        if post.id not in self.id_to_post:
            self.id_to_post[post.id] = post

        for token in post.term_frequencies:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(post.id)

    def get_document_frequency(self, token: str) -> int:
        return len(self.index.get(token, set()))

    def get_inverse_document_frequency(self, token) -> float:
        return math.log10(len(self.id_to_post) / self.get_document_frequency(token))

    def _get_occurrences(self, processed_query: list[str]) -> list[set[str]]:
        return [self.index.get(token, set()) for token in processed_query]

    def _search_tf_idf_cosine_similarity(self, processed_query: list[str]) -> list[PostData]:
        # Convert the query to a vector
        query_vector = self.vectorizer.transform([' '.join(processed_query)]).toarray().reshape(
            self.tf_idf.shape[0], )

        # Calculate the cosine similarity
        similarities = {}
        for i in range(self.tf_idf.shape[1]):
            with np.errstate(all='ignore'):
                similarities[i] = np.dot(self.tf_idf.loc[:, i].values, query_vector) / \
                                  np.linalg.norm(self.tf_idf.loc[:, i]) * np.linalg.norm(query_vector)

        # Sort the values
        similarities_sorted = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        matched_posts = [self.posts[i] for (i, sim) in similarities_sorted if sim > 0]
        return matched_posts

    @timing
    def search(self, query: str, search_type: SearchType = SearchType.AND,
               rank_type: Optional[RankType] = None) -> list[str]:
        """
        Searches for posts containing (all or at least one of) the query terms
        based on cosine similarity of TF-IDF scores, and ranks the results
        based on the given criterion if provided.

        Parameters:
          - query: the query string
          - search_type: (AND, OR) whether all query terms have to match, or just one
          - rank_type: (None, TF_IDF, LIKE_COUNT, COMMENT_COUNT, TAKEN_AT)
                       if set, rank the results based on the given criterion
        """
        print(f'Searching for "{query}"...')

        processed_query = process(query)
        occurrences = self._get_occurrences(processed_query)
        matched_posts = self._search_tf_idf_cosine_similarity(processed_query)
        common_ids = set.intersection(*occurrences)
        common_posts = [post for post in matched_posts if post.id in common_ids]

        if search_type == SearchType.AND:
            matched_posts = common_posts
        if search_type == SearchType.OR:
            matched_posts = common_posts + [post for post in matched_posts if post.id not in common_ids]

        if rank_type == RankType.LIKE_COUNT:
            matched_posts.sort(key=lambda x: x.like_count, reverse=True)
        elif rank_type == RankType.COMMENT_COUNT:
            matched_posts.sort(key=lambda x: x.comment_count, reverse=True)
        elif rank_type == RankType.TAKEN_AT:
            matched_posts.sort(key=lambda x: x.taken_at, reverse=True)

        matched_post_ids = [post.id for post in matched_posts]
        print(f'Found {len(matched_post_ids)} result(s):')
        for i, post_id in enumerate(matched_post_ids):
            print(f'({i + 1}) https://instagram.com/p/{post_id}')

        return matched_post_ids

    @timing
    def search_v1(self, query: str, search_type: SearchType = SearchType.AND,
               rank_type: Optional[RankType] = None) -> list[str]:
        """
        Returns posts containing (all or at least one of) the query terms,
        and ranks them based on the given criterion if provided.

        Parameters:
          - query: the query string
          - search_type: (AND, OR) whether all query terms have to match, or just one
          - rank_type: (None, TF_IDF, LIKE_COUNT, COMMENT_COUNT, TAKEN_AT)
                       if set, rank the results based on the given criterion
        """
        processed_query = process(query)
        results = self._get_occurrences(processed_query)
        matched_posts = []
        if search_type == SearchType.AND:
            # all tokens must be in the post
            matched_posts = [self.id_to_post[post_id] for post_id in set.intersection(*results)]
        if search_type == SearchType.OR:
            # only one token has to be in the post
            matched_posts = [self.id_to_post[post_id] for post_id in set.union(*results)]

        if rank_type == RankType.TF_IDF:
            self.rank_by_tf_idf(processed_query, matched_posts)
        elif rank_type == RankType.LIKE_COUNT:
            matched_posts.sort(key=lambda x: x.like_count, reverse=True)
        elif rank_type == RankType.COMMENT_COUNT:
            matched_posts.sort(key=lambda x: x.comment_count, reverse=True)
        elif rank_type == RankType.TAKEN_AT:
            matched_posts.sort(key=lambda x: x.taken_at, reverse=True)
        return [post.id for post in matched_posts]

    def rank_by_tf_idf(self, processed_query: list[str], posts: list[PostData]) -> None:
        scores = {}
        for post in posts:
            score = 0
            for token in processed_query:
                tf = post.get_term_frequency(token)
                idf = self.get_inverse_document_frequency(token)
                score += tf * idf
            scores[post.id] = score
        posts.sort(key=lambda x: scores[x.id], reverse=True)
