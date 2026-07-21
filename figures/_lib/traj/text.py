"""Text helpers ported verbatim from the paper's normalize_zh.py + tokenize_mixed.py.

``to_simplified`` normalises Chinese to Simplified script (the BabyLM-zho corpus
is Hans); ``tokenize`` is a hybrid CJK/Latin word tokenizer that routes CJK runs
through jieba and everything else through a Unicode-aware word regex, so Dutch
diacritics and English apostrophes survive.

Requires ``opencc`` and ``jieba`` (NOT in the base requirements.txt — install
them for the Fig-7 pipeline: ``pip install opencc-python-reimplemented jieba``).
"""
import re
import unicodedata

_CJK_RUN = re.compile(
    r"[㐀-䶿一-鿿豈-﫿　-〿＀-￯]+"
)
_WORD = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


def _lazy_opencc():
    import opencc
    return opencc.OpenCC("t2s")


_t2s = None


def to_simplified(text: str) -> str:
    """Convert Traditional -> Simplified Chinese (lazy opencc init)."""
    global _t2s
    if _t2s is None:
        _t2s = _lazy_opencc()
    return _t2s.convert(text)


_jieba_ready = False


def _ensure_jieba():
    global _jieba_ready
    if not _jieba_ready:
        import jieba
        jieba.setLogLevel(60)  # suppress init logging
        _jieba_ready = True


def tokenize(text: str) -> list:
    """Return word tokens in order, dropping whitespace/punctuation."""
    import jieba
    _ensure_jieba()
    text = unicodedata.normalize("NFKC", text)  # fullwidth digits/punct -> ascii
    tokens = []
    pos = 0
    for m in _CJK_RUN.finditer(text):
        if m.start() > pos:
            tokens.extend(_WORD.findall(text[pos:m.start()]))
        tokens.extend(
            t for t in jieba.lcut(m.group()) if any(c.isalnum() for c in t)
        )
        pos = m.end()
    if pos < len(text):
        tokens.extend(_WORD.findall(text[pos:]))
    return tokens
