from cryptor.language_tool_detector import LanguageToolDetector
from cryptor.probability_detector import ProbabilityDetector
from cryptor.uncertainty_manipulation import UncertaintyManipulation
from cryptor.regex_detector import RegexDetector
import sys
import json

ALL_CRYPTOR = [
    ProbabilityDetector(n_threads=12),
    RegexDetector(),
    LanguageToolDetector(),
    UncertaintyManipulation(n_threads=12),
]


def hide(feed: list[str], secret: str) -> list[str]:
    return UncertaintyManipulation(n_threads=12).hide(feed, secret).modified_feed


def reconstruct(feed: list[str]) -> str:
    return UncertaintyManipulation(n_threads=12).detect(feed).reconstructed_secret


def detect(feed: list[str]) -> bool:
    detected_something = []
    for detector in ALL_CRYPTOR:
        try:
            contains_secret = detector.detect(feed).contains_secret
        except:
            contains_secret = False

        detected_something.append(contains_secret)
        if contains_secret:
            break

    return any(detected_something)


if __name__ == "__main__":
    input_data = json.load(sys.stdin)

    IS_DETECTOR = False

    if IS_DETECTOR:
        detected_something = detect(input_data['feed'])

        print(json.dumps({'result': detected_something}))
        exit(0)

    elif 'secret' in input_data:
        modified_feed = hide(input_data['feed'], input_data['secret'])

        print(json.dumps({'feed': modified_feed}))
        exit(0)

    else:
        recovered_token = reconstruct(input_data['feed'])

        print(json.dumps({'secret': recovered_token}))
        exit(0)
