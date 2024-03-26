from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectorStatistics:
    pass


@dataclass
class DetectorOutput:
    contains_secret: bool
    reconstructed_secret: Optional[str]
    statistics: DetectorStatistics


class Detector(ABC):
    @abstractmethod
    def detect(self, news_feed: list[str]) -> DetectorOutput:
        pass
