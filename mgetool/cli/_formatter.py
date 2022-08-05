# -*- coding: utf-8 -*-

# @Time  : 2022/8/5 20:55
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
import argparse
import textwrap


class Formatter(argparse.HelpFormatter):
    """Improved help formatter."""

    def _fill_text(self, text, width, indent):
        assert indent == ''
        out = ''
        blocks = text.split('\n\n')
        for block in blocks:
            if block != "":
                if block[0] == '*':
                    # List items:
                    for item in block[2:].split('\n* '):
                        out += textwrap.fill(item,
                                             width=width - 2,
                                             initial_indent='* ',
                                             subsequent_indent='  ') + '\n'
                elif block[0] == ' ':
                    # Indented literal block:
                    out += block + '\n'
                else:
                    # Block of text:
                    out += textwrap.fill(block, width=width) + '\n'
                out += '\n'
            else:
                pass
        return out[:-1]