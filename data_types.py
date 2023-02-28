from dataclasses import dataclass
from datetime import datetime
from enum import Enum


@dataclass
class PostData:
    id: str
    term_frequencies: dict[str, int]
    processed_text: str
    taken_at: datetime
    comment_count: int
    like_count: int

    def get_term_frequency(self, term) -> int:
        return self.term_frequencies.get(term, 0)


class SearchType(Enum):
    AND = 1
    OR = 2


class RankType(Enum):
    TF_IDF = 1
    LIKE_COUNT = 2
    COMMENT_COUNT = 3
    TAKEN_AT = 4
