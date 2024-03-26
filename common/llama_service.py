from typing import Generator
from llama_cpp import Llama
from dataclasses import dataclass
import numpy as np
import nltk


@dataclass
class Choice:
    text: str
    index: int
    logprobs: list[float]
    finish_reason: str


@dataclass
class LlamaOutput:
    prompt: str
    input_chunk: str
    choices: list[Choice]
    prompt_tokens: int
    completion_tokens: int


class LlamaService:
    def __init__(self,
                 llama_model_path: str = "/root/llama-2-7b.Q5_K_M.gguf",
                 n_threads: int = 6,
                 context_size: int = 512,
                 llama_seed: int = 1337,
                 n_gpu_layers: int = 0,
                 ) -> None:
        self.__llama_model_path = llama_model_path

        self.__context_size = context_size
        # TODO: maybe figure out a way hide the seed in the text
        self.llama = Llama(self.__llama_model_path, seed=llama_seed, verbose=False,
                           logits_all=True, n_threads=n_threads, n_ctx=context_size, n_gpu_layers=n_gpu_layers)

    def prompt(self, prompt: str, **kwargs) -> LlamaOutput:
        completion = self.llama(prompt, **kwargs)

        return LlamaOutput(
            prompt=prompt,
            prompt_tokens=completion['usage']['prompt_tokens'],
            completion_tokens=completion['usage']['completion_tokens'],
            input_chunk=prompt,
            choices=[Choice(**choice) for choice in completion['choices']],
        )

    def get_perplexity(self, prompt: str) -> tuple[float, list[float]]:
        tokens = self.llama.tokenize(prompt.encode())
        if len(tokens) < self.__context_size:
            return self.__get_perplexity_on_fly(prompt, 1)

        return self.__get_perplexity_from_chunks(prompt)

    def __get_perplexity_on_fly(self, prompt: str, len_prompt: int) -> tuple[float, list[float]]:
        logits = np.asarray(self.llama(prompt, max_tokens=1, temperature=0, logprobs=1, echo=True)[
                            "choices"][0]["logprobs"][len_prompt + 1:-1])

        return np.exp(-np.mean(np.log(logits))), logits

    def __get_perplexity_from_chunks(self, prompt: str) -> float:
        words = nltk.word_tokenize(prompt)
        step_size = self.__context_size // 2

        tokens_probs = []
        for text_chunk in (words[i:i + step_size] for i in range(0, len(words), step_size // 2)):
            _, probs = self.__get_perplexity_on_fly(
                ' '.join(text_chunk), len_prompt=len(text_chunk) - step_size)

            tokens_probs.extend(probs)

        return np.exp(-np.mean(np.log(tokens_probs))), tokens_probs

    def chunk_data(self, promt: str, data: str, **kwargs) -> list[LlamaOutput]:
        tokens_prompt = len(self.llama.tokenize(promt))

        if tokens_prompt > self.__context_size:
            raise ValueError(
                "Trying to pass a promt bigger than the context size!")

        chunk_completions = []
        for text_chunk in self.__clever_split(data, tokens_prompt):
            completion = self.llama(f"{promt}\n{text_chunk}", **kwargs)

            chunk_completions.append(
                LlamaOutput(
                    prompt=promt,
                    prompt_tokens=completion['usage']['prompt_tokens'],
                    completion_tokens=completion['usage']['completion_tokens'],
                    input_chunk=text_chunk,
                    choices=[Choice(**choice)
                             for choice in completion['choices']],
                )
            )

        return chunk_completions

    def __clever_split(self, data: str, prompt_length: int) -> Generator[str, None, None]:
        total_tokens = len(self.llama.tokenize(data))
        available_content_size = self.__context_size - prompt_length

        if total_tokens < available_content_size:
            yield data
            return

        sentence_split = data.split('.')
        sentence_tokens = [self.llama.tokenize(
            sentence) for sentence in sentence_split]

        yield_data = []
        sentence_context_size_accumulator = 0
        for i in range(len(sentence_split)):
            sentence_tokens = sentence_tokens[i]

            if not yield_data and sentence_tokens > available_content_size:
                words = sentence_split[i].split(' ')
                tokend_per_word = [len(self.llama.tokenize(word))
                                   for word in words]

                yield_words = []
                word_context_size_accumulator = 0
                for j in range(len(words)):
                    yield_words.append(words[j])
                    word_context_size_accumulator += tokend_per_word[j]

                    if (len(words) > j+1 and word_context_size_accumulator + words[j+1] > available_content_size) \
                            or len(words) == j - 1:
                        yield ' '.join(yield_words)
                        yield_words = []
                        word_context_size_accumulator = 0

            yield_data.append(sentence_split[i])
            sentence_context_size_accumulator += sentence_tokens

            if (len(sentence_split) > i+1 and sentence_context_size_accumulator + sentence_tokens[i + 1] > available_content_size) \
                    or len(sentence_split) == i - 1:
                yield ' '.join(yield_data)
                yield_data = []
                sentence_context_size_accumulator = 0
