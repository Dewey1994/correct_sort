from pypinyin import lazy_pinyin, pinyin, Style
import numpy as np

style = Style.TONE3


def get_single_text_pinyin(text: str, heteronym: bool):
    if heteronym:
        py_list = pinyin(text, style=style, heteronym=True)
    else:
        py_list = pinyin(text, style=style, heteronym=False)
    max_size = len(max(py_list, key=len))
    py_list_matrix = []
    for pys in py_list:
        if len(pys) == max_size:
            py_list_matrix.append(pys)
        else:
            py_list_matrix.append(pys * max_size)

    return list(map(list, zip(*py_list_matrix)))


def get_batch_text_pinyin(texts: [str], heteronym: bool):
    res = []
    for text in texts:
        res.append(get_single_text_pinyin(text,heteronym))
    return res