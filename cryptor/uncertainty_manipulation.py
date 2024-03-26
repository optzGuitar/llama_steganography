from typing import Optional
from common.constants import TOKEN_MANIPULATOR_PROMPT_CONFIG
from common.llama_service import LlamaService
from common.uncertainty_service import UncertaintyParameters, UncertaintyService
from cryptor.hider import Hider, HiderOutput
from cryptor.detector import Detector, DetectorOutput
import numpy as np
import base64


class UncertaintyManipulation(Hider, Detector):
    SEC_PREAMBLE_LENGTH = 16

    def __init__(self, n_threads: int = 7, len_memory: int = 3, debug: bool = False) -> None:
        super().__init__()

        self._llama_service = LlamaService(
            context_size=256, n_threads=n_threads)
        self._uncertainty_service = UncertaintyService(
            self._llama_service, UncertaintyParameters())
        self._memory_length = len_memory

        self._debug = debug

    def hide(self, news_feed: list[str], secret: str) -> HiderOutput:
        secret_bits = [int(bit) for byte in base64.b64decode(secret)
                       for bit in f'{byte:08b}']
        num_bits = len(secret_bits)
        length_bits = [int(j) for j in f"{num_bits:016b}"]
        secret_bits = length_bits + secret_bits
        self.log(num_bits)
        self.log(secret_bits)

        all_memories = []

        articles = []
        for article in news_feed:
            article = self._uncertainty_service.filter_unwanted(article)
            sentences = article.split(".")
            sentence_memory = []
            generated_sentences = []

            if not secret_bits:
                articles.append(article)
                continue

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if not secret_bits:
                    generated_sentences.append(sentence)
                    continue

                is_split, first, _ = self.split_sentences(sentence)
                if not is_split:
                    continue

                sentence_memory.append(first)

                if len(sentence_memory) > self._memory_length:
                    sentence_memory.pop(0)

                all_memories.append(sentence_memory.copy())

                self.log(". ".join(sentence_memory))

                output = self._llama_service.prompt(
                    ". ".join(sentence_memory),
                    logits_processor=lambda _, x: self._manipulate(
                        secret_bits, x),
                    ** TOKEN_MANIPULATOR_PROMPT_CONFIG
                )

                generated_sentence = f"{first}{output.choices[0].text}"
                sentence_memory.pop(-1)
                sentence_memory.append(generated_sentence)
                generated_sentences.append(generated_sentence)

            new_article = ". ".join(generated_sentences)
            new_article += "."
            articles.append(new_article)

        return HiderOutput(
            modified_feed=articles,
            statistics=all_memories
        )

    def split_sentences(self, text: str) -> tuple[bool, str, str]:
        split_location = 4
        words = text.split(" ")
        num_words = len(words)

        if num_words <= split_location:
            return False, text, ""

        second = " ".join(words[split_location:])

        if not second.endswith("."):
            second += "."

        return True, " ".join(words[:split_location]) + " ", second

    def _manipulate(self, secret_bits: list[bool], logits: np.ndarray) -> np.ndarray:
        if not secret_bits:
            return self._force_no_multibyte(logits)

        copy = logits.copy()
        is_applicable, max_tokens, available_tokens = self._uncertainty_service.is_applicable(
            copy)

        if not is_applicable:
            return self._force_no_multibyte(logits)

        try:
            bits = []
            for bit in range(max_tokens):
                if len(secret_bits) <= bit:
                    break

                bits.append(secret_bits[bit])
                index = int(''.join((str(int(i)) for i in bits)), 2)

                if index >= max_tokens:
                    bits.pop(-1)
                    break

            if bits[0] == 0 and len(bits) > 1:
                bits = bits[:1]
            for _ in range(len(bits)):
                secret_bits.pop(0)

            index = int(''.join((str(int(i)) for i in bits)), 2)
            logits = self._select_token(logits, available_tokens, index)

        except UnicodeError:
            pass

        return logits

    def _select_token(self, logits: np.ndarray, available_tokens: list[int], index: int) -> np.ndarray:
        self.log(index, len(available_tokens),
                 self._llama_service.llama.detokenize([available_tokens[index]]))

        indices = available_tokens[:index]
        logits[indices] = -1000
        logits[available_tokens[index]] = 1000

        return logits

    def detect(self, news_feed: list[str]) -> DetectorOutput:
        secret_bits = []
        stop_p = [False]
        len_bits = None

        for article in news_feed:
            sentences = article.split(".")
            sentence_memory = []

            for sentence in sentences:
                sentence = sentence.lstrip()
                if not sentence:
                    continue

                if len_bits is None and len(secret_bits) >= 16:
                    len_bits = int(''.join((str(int(i))
                                   for i in secret_bits[:UncertaintyManipulation.SEC_PREAMBLE_LENGTH])), 2)
                    secret_bits = secret_bits[UncertaintyManipulation.SEC_PREAMBLE_LENGTH:]

                if len_bits is not None and len(secret_bits) >= len_bits:
                    stop_p[0] = True
                    break

                is_split, first, second = self.split_sentences(sentence)
                if not is_split:
                    raise RuntimeError(
                        "Splitting must always work during decoding!")

                sentence_memory.append(first)

                while len(sentence_memory) > self._memory_length:
                    sentence_memory.pop(0)

                guide = second.split(" ")
                self.log(". ".join(sentence_memory))
                _ = self._llama_service.prompt(
                    ". ".join(sentence_memory),
                    logits_processor=lambda _, x: self._recover_manipulate(
                        secret_bits, guide, stop_p, len_bits, x),
                    ** TOKEN_MANIPULATOR_PROMPT_CONFIG
                )
                sentence_memory.pop(-1)
                sentence_memory.append(sentence.rstrip("."))

                if stop_p[0]:
                    break

            if stop_p[0]:
                secret_bits = secret_bits[:len_bits]
                break

        self.log(secret_bits)
        reconstruction = self.bits_to_bytes(secret_bits)

        return DetectorOutput(
            contains_secret=bool(reconstruction),
            reconstructed_secret=reconstruction,
            statistics=None
        )

    def _recover_manipulate(self, secret_bits: list[bool], sentence: list[str], stop_p: list[bool], len_bits: Optional[int], logits: np.ndarray) -> np.ndarray:
        if stop_p[0]:
            return logits

        if len_bits is not None and len(secret_bits) >= len_bits:
            stop_p[0] = True
            return logits

        if not ''.join(sentence):
            self.log("Warning: empty string")
            return logits

        copy = logits.copy()
        is_applicable, max_tokens, available_tokens = self._uncertainty_service.is_applicable(
            copy)

        if not is_applicable:
            logits = self._force_no_multibyte(logits)
            token = self._uncertainty_service.sample_from_llama(logits)
            decoded = self._llama_service.llama.detokenize(
                [token]).decode()

            if decoded == "":
                return logits

            self._uncertainty_service.update_sentence(sentence, decoded)

            return logits

        pos, _ = self._uncertainty_service.indentify_selected_token(
            sentence, available_tokens
        )

        if pos is None:
            raise RuntimeError("Could not identify selected token!")

        secret_bits.extend([int(i) for i in f"{pos:b}"])

        return self._select_token(logits, available_tokens, pos)

    def _force_no_multibyte(self, logits: np.ndarray) -> np.ndarray:
        while True:
            token = self._uncertainty_service.sample_from_llama(logits)
            try:
                decoded = self._llama_service.llama.detokenize(
                    [token]).decode()
            except UnicodeDecodeError:
                logits[token] = -1000
                continue

            break

        return logits

    def log(self, *args) -> None:
        if self._debug:
            print(*args)

    def bits_to_bytes(self, bits_list) -> str:
        bytes_data = bytearray()
        for i in range(0, len(bits_list), 8):
            byte_bits = bits_list[i:i+8]
            byte_value = int(''.join(map(str, byte_bits)), 2)
            bytes_data.append(byte_value)
        return base64.b64encode(bytes(bytes_data)).decode()
