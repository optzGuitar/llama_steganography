import re
from common.constants import ABREVIATIONS_THRESHOLD_PERCENT, DOUBLED_PUNCTUATION_THRESHOLD_PERCENT, NUMBER_THRESHOLD_PERCENT, SPECIAL_CHARACTERS
from cryptor.detector import Detector, DetectorOutput, DetectorStatistics
import pandas as pd
import string


class RegexDetector(Detector):
    def __init__(self) -> None:
        self._rgx = {
            re.compile(r"(\d+,{0,1})+(.\d+){0,1}"): NUMBER_THRESHOLD_PERCENT,
            re.compile(r"[A-Z]{2,5}"): ABREVIATIONS_THRESHOLD_PERCENT,
            re.compile("(" + "|".join(string.whitespace + re.escape(string.punctuation) + ''.join(SPECIAL_CHARACTERS)) + ")"): DOUBLED_PUNCTUATION_THRESHOLD_PERCENT,
        }

        self._data = pd.DataFrame(columns=range(len(self._rgx) + 1))

    def detect(self, news_feed: list[str]) -> DetectorOutput:
        for news in news_feed:
            new_row = [len(news)]
            for rgx in self._rgx:
                new_row.append(len(rgx.findall(news)))

            self._data.loc[len(self._data)] = new_row

        found_something = False
        for col, threshold in zip(self._data.columns[1:], self._rgx.values()):
            normalized_col = self._data[col] / self._data[0]
            found_something |= (normalized_col > threshold).any()
            found_something |= (
                self._data[col].sum() / self._data[0].sum() > threshold
            )

        return DetectorOutput(
            contains_secret=found_something,
            reconstructed_secret="",
            statistics=DetectorStatistics(),
        )
