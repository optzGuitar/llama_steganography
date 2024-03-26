TOKEN_MANIPULATOR_LLAMA_SEED = 6397
TOKEN_MANIPULATOR_CONTEXT_SIZE = 1024
TOKEN_MANIPULATOR_DISCOUNT_FACTOR = 0.5
TOKEN_MANIPULATOR_PROMPT_CONFIG = {
    "echo": False,
    "max_tokens": 256,
    "temperature": 0,
    "top_p": 0.0,
    "top_k": 0,
    "stop": ['.', "?", "!"],
    "repeat_penalty": 1.5,
}
TOKEN_MANIPULATOR_SENTENCE_CONDITION_CUTOFF = 6

TOKEN_BIT_DILUTION_FACTOR: int = 1

NUMBER_THRESHOLD_PERCENT = 0.1
ABREVIATIONS_THRESHOLD_PERCENT = 0.09
DOUBLED_PUNCTUATION_THRESHOLD_PERCENT = 1.017
LANGUAGE_TOOL_THRESHOLD_PERCENT = 0.03

SPECIAL_CHARACTERS = ['\x0b',
                      '\x0c',
                      '\x85',
                      '\xa0',
                      '\u1680',
                      '\u2000',
                      '\u2001',
                      '\u2002',
                      '\u2003',
                      '\u2004',
                      '\u2005',
                      '\u2006',
                      '\u2007',
                      '\u2008',
                      '\u2009',
                      '\u200a',
                      '\u2028',
                      '\u2029',
                      '\u202f',
                      '\u205f',
                      '\u3000']
