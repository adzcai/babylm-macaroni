"""Text helpers for corpus mining: zh normalization, mixed CJK/Latin
tokenization, function-word lists, and script-consistency checks.

Ported verbatim (behaviour-preserving) from the paper's ``normalize_zh.py``,
``tokenize_mixed.py``, ``stopwords.py``, and ``lang_script.py``. The three
third-party dependencies (``opencc``, ``jieba``, ``stopwordsiso``) are imported
lazily so that the *measurement* half of the pipeline — which never touches raw
text — imports cleanly without them installed.
"""
import re
import unicodedata

LANGS = ("eng", "nld", "zho")

# --------------------------------------------------------------------------- #
# Chinese Traditional -> Simplified (the zho corpus is Simplified; the CS
# generator emits some Traditional text, so both sides are normalized first).
# --------------------------------------------------------------------------- #
_T2S = None


def to_simplified(text: str) -> str:
    global _T2S
    if _T2S is None:
        import opencc  # lazy: only corpus mining needs it
        _T2S = opencc.OpenCC("t2s")
    return _T2S.convert(text)


def norm_word(w: str) -> str:
    return to_simplified(w.lower())


# --------------------------------------------------------------------------- #
# Hybrid CJK/Latin word tokenizer (jieba for CJK runs, Unicode-aware regex for
# the rest, stitched back in order). Handles Dutch diacritics + apostrophes.
# --------------------------------------------------------------------------- #
_CJK_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿　-〿＀-￯]+")
_WORD = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)
_JIEBA = None


def _jieba():
    global _JIEBA
    if _JIEBA is None:
        import jieba
        jieba.setLogLevel(60)
        _JIEBA = jieba
    return _JIEBA


def tokenize(text: str) -> list:
    """Return word tokens in order, dropping whitespace/punctuation."""
    text = unicodedata.normalize("NFKC", text)  # fullwidth digits/punct -> ascii
    tokens = []
    pos = 0
    for m in _CJK_RUN.finditer(text):
        if m.start() > pos:
            tokens.extend(_WORD.findall(text[pos:m.start()]))
        tokens.extend(t for t in _jieba().lcut(m.group()) if any(c.isalnum() for c in t))
        pos = m.end()
    if pos < len(text):
        tokens.extend(_WORD.findall(text[pos:]))
    return tokens


# --------------------------------------------------------------------------- #
# Closed-class (function) word lists via stopwordsiso, + a few it misses.
# --------------------------------------------------------------------------- #
_ISO = {"eng": "en", "nld": "nl", "zho": "zh"}
_EXTRA = {
    "nld": {"eenmaal", "hebbend", "meeste", "zeer"},
    "zho": {"应该", "没有", "没", "非常", "很多", "同样", "嗎"},
    "eng": set(),
}
_STOPWORDS = None


def _stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        import stopwordsiso as _sw
        _STOPWORDS = {
            lang: {w.lower() for w in _sw.stopwords(iso)} | _EXTRA.get(lang, set())
            for lang, iso in _ISO.items()
        }
    return _STOPWORDS


def is_function_word(word: str, lang: str) -> bool:
    return word.strip().lower() in _stopwords().get(lang, set())


# --------------------------------------------------------------------------- #
# Script-consistency check for mined pairs (drop Latin-as-Chinese artifacts).
# --------------------------------------------------------------------------- #
def _has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" or "㐀" <= c <= "䶿"
               or "豈" <= c <= "﫿" for c in s)


def _has_latin(s: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in s)


def word_in_script(word: str, lang: str) -> bool:
    """True if ``word`` is written in ``lang``'s expected script."""
    if lang == "zho":
        return _has_cjk(word) and not _has_latin(word)
    return _has_latin(word) and not _has_cjk(word)


def pair_script_ok(embedded_word, embedded_lang, replaced_word, matrix_lang) -> bool:
    return (word_in_script(embedded_word, embedded_lang)
            and word_in_script(replaced_word, matrix_lang))
