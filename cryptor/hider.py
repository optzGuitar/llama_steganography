from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class HiderStatistics:
    pass


@dataclass
class HiderOutput:
    modified_feed: list[str]
    statistics: HiderStatistics


class Hider(ABC):
    @abstractmethod
    def hide(self, news_feed: list[str], secret: str) -> HiderOutput:
        pass
