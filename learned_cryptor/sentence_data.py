import pickle
from typing import Any
import pandas as pd
import torch
from torch.utils.data import Dataset
import base64
from torchtext.data import get_tokenizer
import string
import io
from torchtext.vocab import Vocab
import numpy as np


class Tokenizer:
    def __init__(self, context_size: int):
        self.start_token = 0
        self.end_token = 1
        self.pad_token = 2
        self.unk_token = 3
        self._offset = self.unk_token + 1

        self._chars = string.digits + string.ascii_letters + string.punctuation
        self.vocab = {
            c: i + self._offset for i, c in enumerate(self._chars)
        }

        self._context_size = context_size

    def vocab_length(self):
        return len(self.vocab) + self._offset

    def __call__(self, secret: str, add_end: bool, add_start: bool) -> torch.Tensor:
        tokens = [self.start_token] if add_start else []
        tokens += [self.vocab[c] for c in secret if c in self._chars]
        tokens += [self.end_token] if add_end else []

        return torch.as_tensor(self._pad(tokens))

    def _pad(self, tokens: list) -> list:
        if len(tokens) > self._context_size:
            return tokens[:self._context_size]
        return tokens + [self.pad_token] * (self._context_size - len(tokens))

    def decode(self, tokens: list) -> str:
        try:
            end = tokens.index(self.end_token)
            tokens = tokens[:end]
        except ValueError:
            pass

        keys = list(self.vocab.keys())
        return ''.join([keys[i - self._offset] for i in tokens if i >= self._offset])


class SentenceData(Dataset):
    def __init__(self, context_size: int, secret_len_range=(1, 4)):
        self._sentences = pd.read_pickle(
            'sentence_perplexity_data_cleaned_en_50000.pkl')
        self._secret_len_range = secret_len_range

        with open('vocab.pkl', 'rb') as f:
            self._tokenizer = pickle.load(f)

        self._secret_tokenizer = Tokenizer(context_size)
        self.context_size = context_size

    def __len__(self):
        return len(self._sentences)

    def vocab_length(self):
        return 15000

    def secret_vocab_length(self):
        return self._secret_tokenizer.vocab_length()

    def __getitem__(self, idx):
        row = self._sentences.iloc[idx]

        # perplexity = row['perplexity']
        sentence = row['sentence']
        secret, secret_inp, secret_target = self.get_sec_src_inp_tar()

        sentence, sentence_inp, sentence_inp_half = self.get_src_inp_tar(
            sentence)

        return sentence, sentence_inp, sentence_inp_half, secret, secret_inp, secret_target

    def get_sec_src_inp_tar(self):
        secret_ids = torch.randint(
            0, len(self._secret_tokenizer._chars), (1,)).item()
        secret = ''.join(self._secret_tokenizer._chars[secret_ids])

        secret_target = secret[1:]
        secret_inp = secret[:-1]

        secret_target = self._secret_tokenizer(secret_target, True, False)
        secret = self._secret_tokenizer(secret, True, True)
        secret_inp = self._secret_tokenizer(secret_inp, False, True)

        return secret, secret_inp, secret_target

    def get_src_inp_tar(self, sentence: str):
        start_tok = self._tokenizer(["<start>"])
        eos_tok = self._tokenizer(["<eos>"])
        sentence = sentence.split(" ")
        sentence_tok = start_tok + self._tokenizer(sentence) + eos_tok
        sentence_inp = start_tok + self._tokenizer(sentence)
        sentence_inp_half = start_tok + \
            self._tokenizer(sentence[:6]
                            )

        sentence_tok = self._pad(sentence_tok)
        sentence_inp = self._pad(sentence_inp)
        sentence_inp_half = self._pad(sentence_inp_half)

        return torch.as_tensor(sentence_tok), torch.as_tensor(sentence_inp), torch.as_tensor(sentence_inp_half)

    def _pad(self, tokens: list) -> list:
        if len(tokens) > self.context_size:
            return tokens[len(tokens) - self.context_size:]
        return tokens + [2] * (self.context_size - len(tokens))
