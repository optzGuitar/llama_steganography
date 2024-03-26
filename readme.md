# Steganography with Llama2
In this project, I explored hiding information in text using large language models and how to detect if text was manipulated.

## Usage

The [main.py](main.py) file can be executed directly from the command line. It accepts JSON input through the standard input (stdin), which should contain a "feed" key with a list of strings. Depending on the operation mode, the input may also need to contain a "secret" key for the hiding operation. The secret is expected to be base64 encoded.

```
{
  "feed": [
    "Today is a beautiful day to learn something new!",
    "I've been exploring Python and its vast libraries.",
    "Using LLMs to hide information is fun."
  ],
  "secret": "TWVldCBtZSBhdCB0aGUgY2FmZSBhdCAzIFBNLg=="
}
```

The secret key should be omitted when trying to decrypt information and when detecting manipulated texts. To enable the detector, set IS_DETECTOR in the main.py to True.


## Experimental

The [learned_cryptor](learned_cryptor/) contains a FiLM[1] & Transformer[2] combination to learn the hiding and decrypting. The training can be started using [train_model.py](train_model.py)

# References
[1] Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). Film: Visual reasoning with a general conditioning layer. *Proceedings of the AAAI Conference on Artificial Intelligence*. [https://arxiv.org/abs/1709.07871](https://arxiv.org/abs/1709.07871)

[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
