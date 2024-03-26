from common.constants import LANGUAGE_TOOL_THRESHOLD_PERCENT
from cryptor.detector import Detector, DetectorOutput, DetectorStatistics
from language_tool_python import LanguageTool
import numpy as np


class LanguageToolDetector(Detector):
    def __init__(self):
        self.language_tools = [
            LanguageTool('en-US'),
            LanguageTool('de-DE'),
            LanguageTool('fr-FR'),
            LanguageTool('it-IT'),
            LanguageTool('el-GR'),
            LanguageTool('es-ES'),
            LanguageTool('nl-NL'),
            LanguageTool('pt-PT'),
            LanguageTool('ru-RU'),
            LanguageTool('zh-CN'),
            LanguageTool('ja-JP'),
        ]

    def detect(self, news_feed: list[str]) -> DetectorOutput:
        num_matches = []
        len_article = []

        for news in news_feed:
            num_matches.append(self.check_grammar(news))
            len_article.append(len(news))

        num_matches = np.asarray(num_matches)
        len_article = np.asarray(len_article)

        percentages = num_matches / len_article
        total_percentage = num_matches.sum() / len_article.sum()

        found_something = (percentages > LANGUAGE_TOOL_THRESHOLD_PERCENT).any()
        found_something |= total_percentage > LANGUAGE_TOOL_THRESHOLD_PERCENT

        return DetectorOutput(
            contains_secret=found_something,
            reconstructed_secret="",
            statistics=DetectorStatistics(),
        )

    def get_trigger_word(self, text, error):
        offset = error.offset
        error_length = error.errorLength
        return text[offset:offset + error_length]

    def check_grammar(self, text):
        matches = self.language_tools[0].check(text)
        filtered_matches = []

        for match in matches:
            for tool in self.language_tools[1:]:
                if match.ruleId == 'MORFOLOGIK_RULE_EN_US':
                    word = self.get_trigger_word(text, match)

                    if not tool.check(word):
                        continue

            filtered_matches.append(match)

        return filtered_matches
