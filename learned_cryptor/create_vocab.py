import pandas as pd
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pickle

specials = ["<start>", "<eos>",  "<pad>", "<unk>"]

if __name__ == "__main__":
    sentences = pd.read_pickle(
        '/Users/leopinetzki/Library/Mobile Documents/com~apple~CloudDocs/data/Studium/Semester 9/STEMO/sentence_perplexity_data_cleaned_en_50000.pkl')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        map(tokenizer, sentences['sentence']), specials=specials,
        min_freq=3,
        max_tokens=15000,
    )
    vocab.set_default_index(vocab["<unk>"])

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
