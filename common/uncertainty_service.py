from dataclasses import dataclass
import inspect
from itertools import combinations
import numpy as np
from emoji import EMOJI_DATA, emoji_list

from common.constants import TOKEN_MANIPULATOR_PROMPT_CONFIG
from common.llama_service import LlamaService
import string
from scipy.special import softmax


@dataclass
class UncertaintyParameters:
    probability_threshold: float = 0.6


class UncertaintyService:
    def __init__(self, llama_service: LlamaService, params: UncertaintyParameters, debug: bool = False) -> None:
        self.params = params
        self._llama_service = llama_service

        self._simplified_punktuation = [".", "!", "?"]

        self._emoji_bytes = "".join(list(EMOJI_DATA.keys()))

        self._debug = debug

    def sample_from_llama(self, logits: np.ndarray) -> int:
        return self._llama_service.llama.sample(
            logits_processor=lambda a, b: logits,
            **{k: v for k, v in TOKEN_MANIPULATOR_PROMPT_CONFIG.items()
               if k in inspect.getfullargspec(self._llama_service.llama.sample).args}
        )

    def num_logits_until_threshold(self, logits: np.ndarray) -> int:
        probs = softmax(logits)
        probs[::-1].sort()
        cumsum = np.cumsum(probs)
        mask_under_threshold = cumsum < self.params.probability_threshold

        return mask_under_threshold.sum()

    def _tokens_undistinguishable(self, logits: np.ndarray, n_tokens_to_consider: int) -> tuple[int, list[int]]:
        manipulated_logits = logits.copy()

        all_decoded = []
        sampled_tokens = []
        for _ in range(n_tokens_to_consider):
            sampled_token = self.sample_from_llama(manipulated_logits)
            sampled_tokens.append(sampled_token)
            try:
                token_chars = self._llama_service.llama.detokenize(
                    [sampled_token]
                ).decode()
            except UnicodeDecodeError:
                break

            if token_chars in self._simplified_punktuation[1:]:
                break
            if not all([i in string.printable for i in token_chars]):
                break

            all_decoded.append(token_chars)
            manipulated_logits[manipulated_logits.argmax()] = -1000

        toks_zero = set()
        no_conflict = True
        for tok0, tok1 in combinations(all_decoded, 2):
            toks_zero.add(tok0)
            if len(tok0) > len(tok1):
                tok0, tok1 = tok1, tok0

            if tok1.strip().startswith(tok0.strip()):
                no_conflict = False
                break

        self.log(f"Tokens: {all_decoded[:len(toks_zero) + no_conflict]}")
        return len(toks_zero) + no_conflict, sampled_tokens[:len(toks_zero) + no_conflict]

    def is_applicable(self, logits: np.ndarray) -> tuple[bool, int, list[int]]:
        num_uncertainty_tokens = self.num_logits_until_threshold(logits.copy())
        max_tokens, available_tokens = self._tokens_undistinguishable(
            logits, num_uncertainty_tokens
        )

        if max_tokens > len(available_tokens):
            max_tokens = len(available_tokens)

        return num_uncertainty_tokens > 1 and max_tokens > 1, max_tokens, available_tokens

    def filter_unwanted(self, strng: str) -> str:
        while emoji_list(strng):
            lst = emoji_list(strng)
            pos = lst["match_start"]
            end_pos = lst["match_end"]
            strng = strng[:pos] + strng[end_pos:]

        return "".join([
            i for i in strng if i in string.printable
        ])

    def indentify_selected_token(self, sentence: list[str], available_tokens: list[int]) -> tuple[int, str]:
        complete_sentence = ' '.join(sentence)

        for pos, index in enumerate(available_tokens):
            token = self._llama_service.llama.detokenize(
                [index]
            ).decode()

            token_pre = token
            for repl in self._simplified_punktuation:
                token = token.replace(repl, ".")

            replaced_something = token != token_pre or "." in token_pre

            if complete_sentence.startswith(token) or (replaced_something and token.startswith(complete_sentence)):
                self.update_sentence(sentence, token)
                return pos, sentence

        return None, sentence

    def update_sentence(self, sentence: list[str], token: str) -> list[str]:
        complete_sentence = ' '.join(sentence)
        self.log(token)

        if complete_sentence[0] != " " and token[0] == " ":
            complete_sentence = " " + complete_sentence

        if complete_sentence.lstrip() == "." and (token == "\n" or any([tc in self._simplified_punktuation for tc in token.lstrip()])):
            pass
        elif complete_sentence == "".join([i for i in token if i not in self._simplified_punktuation]):
            pass
        elif not (complete_sentence[:len(token)] == token):
            raise ValueError(
                f"Sentence is {complete_sentence[:len(token)]} but token is {token}"
            )

        complete_sentence = complete_sentence[len(token):]

        sentence.clear()
        sentence.extend(complete_sentence.split(" "))

        return sentence

    def log(self, *args) -> None:
        if self._debug:
            print(*args)
