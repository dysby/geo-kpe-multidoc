import re

from nltk.stem import StemmerI


def clean_keywords(strings, stemmer: StemmerI):
    """from https://github.com/asahi417/kex/"""

    def cleaner(_string):
        _string = _string.lower()
        _string = re.sub(r"\A\s*", "", _string)
        _string = re.sub(r"\s*\Z", "", _string)
        _string = (
            _string.replace("\n", " ")
            .replace("\t", "")
            .replace("-", " ")
            .replace(".", "")
        )
        _string = re.sub(r"\s{2,}", " ", _string)
        _string = " ".join(list(map(lambda x: stemmer.stem(x), _string.split(" "))))
        return _string

    keys = list(
        # filter(lambda x: len(x) > 0, [cleaner(s) for s in re.split("[\n,]", strings)])
        filter(None, [cleaner(s) for s in strings])
    )
    return list(set(keys))


def translate_parentesis(string):
    escaped_punctuation = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }
    for k, v in escaped_punctuation.items():
        string = string.replace(k, v)
    return string
