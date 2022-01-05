# -*- coding: utf-8 -*-

# @Time    : 2021/8/6 15:42
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause


try:
    from translate import Translator
except ImportError:
    raise ImportError("'translate' should be install. Try ``pip install translate``")


def translate_en_to_zh(words: str):
    translator = Translator(to_lang="zh")
    return translator.translate(words)

def remove_empty(lists):
    return [i for i in lists if i != '']


def get_msgid(text_lines):
    text = "".join(text_lines)
    text_ms = text.split("msgid ")
    text_cup = [i.split("msgstr ") for i in text_ms]
    data = []

    for i in text_cup:
        datai=[]
        if len(i)==2:
            str_msgid = remove_empty(i[0].split("\n"))
            str_msgstr = remove_empty(i[1].split("\n"))
            datai.append(str_msgid)
            datai.append(str_msgstr)
        else:
            str_ = remove_empty(i[0].split("\n"))
            datai.append(str_)
        data.append(datai)

    data_new = []
    for data_cup in data:
        if len(data_cup) == 2:
            for i in range(len(data_cup[0])):
                try:
                    new = translate_en_to_zh(data_cup[0][i])
                    data_cup[1][i]=new
                except BaseException:
                    pass
            data_new.append(data_cup)

    return data

# f = open(r"C:\Users\Administrator\PycharmProjects\featurebox\docs\locale\zh_CN\LC_MESSAGES\src\featurebox.selection.po")
# text_ = f.readlines()
# s= get_msgid(text_lines=text_)



