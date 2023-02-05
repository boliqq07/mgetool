# -*- coding: utf-8 -*-

# @Time     : 2021/8/15 14:09
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx

u"""通用画图部分模块
1. 用代码画的图，可以先尝试调用这里的模版画图。
2. 无模版，需要自定义的代码生成的图片，画图代码和图片直接放在自己章节文件夹位置。代码画图推荐使用 plotly, seaborn, networkx, matplotlib.
3. 非代码画图，如使用origin，Matlab, ppt, visio, ps等手动画图，需要上传图片到对应章节！
"""

import re
from itertools import product, zip_longest

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colorbar import ColorbarBase
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import roc_curve, auc
from sklearn.utils import check_random_state
from sympy.physics.quantum.circuitplot import pyplot

markers = ['o', 's', '^', 'v', '<', '>', 'p', 'D', '+', '*', 'H', '1', '2', '3', '4'] * 5
ls = ['-', '-.', '- ', ':'] * 10


def setting_rcParams_single(font="Arial", figure_size=(10, 7.5)):
    """短图通用配置

    本函数无返回值，rcParams为全局变量，调用本函数即可。

    Example:

    >>> setting_rcParams_single() #调用本函数即可
    """

    if font is None or font == "DejaVu Sans":
        pass
    elif font == "Times":
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times new roman']
    elif font == "Arial":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["Arial"]
    elif font == "Helvetica":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["Helvetica"]
    elif font == "Tahoma":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["Tahoma"]
    elif font == "Monospace":
        rcParams['font.family'] = 'Monospace'
    elif font in ["SimHei", "黑体"]:
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["SimHei"]
    elif font in ["SimSun", "宋体"]:
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ["SimSun"]
    elif font == "宋体+Times":
        config = {
            "font.family": 'serif',
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
        }
        print(r"使用 $\mathrm{This \; is \; one \; sample}$ 包裹英文，变为新罗马，否则仍然为宋体，\; 代表空格。（中文勿改）")
        rcParams.update(config)
    elif font == "宋体+DejaVu":
        config = {
            "font.family": 'serif',
            "font.serif": ['SimSun'],
        }
        print(r"使用 $\mathrm{This \; is \; one \; sample}$ 包裹英文，变为新罗马，否则仍然为宋体，\; 代表空格。（中文勿改）")
        rcParams.update(config)
    else:
        NotImplementedError('just accept ["宋体+Times","宋体","黑体","Times","Arial","Tahoma","Helvetica"]')

    rcParams['axes.unicode_minus'] = False
    rcParams['figure.dpi'] = 300
    rcParams['figure.figsize'] = figure_size

    rcParams['figure.subplot.left'] = 0.1
    rcParams['figure.subplot.right'] = 0.92
    rcParams['figure.subplot.top'] = 0.92
    rcParams['figure.subplot.bottom'] = 0.1

    rcParams['axes.titlesize'] = 24
    rcParams['axes.labelsize'] = 24
    rcParams['axes.linewidth'] = 3

    rcParams['xtick.major.width'] = 3
    rcParams['ytick.major.width'] = 3
    rcParams['xtick.major.size'] = 8
    rcParams['ytick.major.size'] = 8

    rcParams['xtick.labelsize'] = 22
    rcParams['ytick.labelsize'] = 22
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'

    rcParams['legend.fontsize'] = 20

    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.15


def setting_rcParams_long(font="Arial", figure_size=(14.5, 11)):
    """长通用配置

    本函数无返回值，rcParams为全局变量，调用本函数即可

    Example:

    >>> setting_rcParams_single() #调用本函数即可
    """

    if font is None or font == "DejaVu Sans":
        pass
    elif font == "Times":
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times new roman']
    elif font == "Arial":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["Arial"]
    elif font == "Helvetica":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["Helvetica"]
    elif font == "Tahoma":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["Tahoma"]
    elif font == "Monospace":
        rcParams['font.family'] = 'Monospace'
    elif font in ["SimHei", "黑体"]:
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["SimHei"]
    elif font in ["SimSun", "宋体"]:
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ["SimSun"]
    elif font == "宋体+Times":
        config = {
            "font.family": 'serif',
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
        }
        print(r"使用 $\mathrm{This \; is \; one \; sample}$ 包裹英文，变为新罗马，否则仍然为宋体，\; 代表空格。（中文勿改）")
        rcParams.update(config)
    elif font == "宋体+DejaVu":
        config = {
            "font.family": 'serif',
            "font.serif": ['SimSun'],
        }
        print(r"使用 $\mathrm{This \; is \; one \; sample}$ 包裹英文，变为新罗马，否则仍然为宋体，\; 代表空格。（中文勿改）")
        rcParams.update(config)
    else:
        NotImplementedError('just accept ["宋体+Times","宋体","黑体","Times","Arial","Tahoma","Helvetica"]')

    rcParams['axes.unicode_minus'] = False
    rcParams['figure.dpi'] = 300
    rcParams['figure.figsize'] = figure_size

    rcParams['figure.subplot.left'] = 0.09
    rcParams['figure.subplot.right'] = 0.93
    rcParams['figure.subplot.top'] = 0.93
    rcParams['figure.subplot.bottom'] = 0.09

    rcParams['axes.titlesize'] = 16
    rcParams['axes.labelsize'] = 16
    rcParams['axes.linewidth'] = 2

    rcParams['xtick.major.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['xtick.major.size'] = 6
    rcParams['ytick.major.size'] = 6

    rcParams['xtick.labelsize'] = 15
    rcParams['ytick.labelsize'] = 15
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'

    rcParams['legend.fontsize'] = 16

    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.15


def setting_rcParams_part(font="Arial", figure_size=(7, 5)):
    """1/2分图中的单个图，通用配置（后续手动再合并为长图）

    本函数无返回值，rcParams为全局变量，调用本函数即可

    Example:

    >>> setting_rcParams_single() #调用本函数即可
    """

    if font is None or font == "DejaVu Sans":
        pass
    elif font == "Times":
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times new roman']
    elif font == "Arial":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["Arial"]
    elif font == "Helvetica":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["Helvetica"]
    elif font == "Tahoma":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["Tahoma"]
    elif font == "Monospace":
        rcParams['font.family'] = 'Monospace'
    elif font in ["SimHei", "黑体"]:
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["SimHei"]
    elif font in ["SimSun", "宋体"]:
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ["SimSun"]
    elif font == "宋体+Times":
        config = {
            "font.family": 'serif',
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
        }
        print(r"使用 $\mathrm{This \; is \; one \; sample}$ 包裹英文，变为新罗马，否则仍然为宋体，\; 代表空格。（中文勿改）")
        rcParams.update(config)
    elif font == "宋体+DejaVu":
        config = {
            "font.family": 'serif',
            "font.serif": ['SimSun'],
        }
        print(r"使用 $\mathrm{This \; is \; one \; sample}$ 包裹英文，变为新罗马，否则仍然为宋体，\; 代表空格。（中文勿改）")
        rcParams.update(config)
    else:
        NotImplementedError('just accept ["宋体+Times","宋体","黑体","Times","Arial","Tahoma","Helvetica"]')

    rcParams['axes.unicode_minus'] = False
    rcParams['figure.dpi'] = 300
    rcParams['figure.figsize'] = figure_size

    rcParams['figure.subplot.left'] = 0.12
    rcParams['figure.subplot.right'] = 0.93
    rcParams['figure.subplot.top'] = 0.93
    rcParams['figure.subplot.bottom'] = 0.12

    rcParams['axes.titlesize'] = 32
    rcParams['axes.labelsize'] = 32
    rcParams['axes.linewidth'] = 3.5

    rcParams['xtick.major.width'] = 3.5
    rcParams['ytick.major.width'] = 3.5
    rcParams['xtick.major.size'] = 8
    rcParams['ytick.major.size'] = 8

    rcParams['xtick.labelsize'] = 30
    rcParams['ytick.labelsize'] = 30
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'

    rcParams['legend.fontsize'] = 26

    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.15


def spilt_chinese_and_english(strings):
    """英文转换为latex"""

    def subs_en(string2):
        uncn = re.compile(r'[^\u4e00-\u9fa5]*')
        string2e = uncn.findall(string2)
        string2e = [i for i in string2e if i != '']
        string2e = [i.replace(" ", " \; ") for i in string2e]
        string2e = ["$\mathrm{{{}}}$".format(i) for i in string2e]

        uncn2 = re.compile(r'[\u4e00-\u9fa5]*')
        string2c = uncn2.findall(string2)
        string2c = [i for i in string2c if i != '']

        string2_ = []
        if string2[0].isascii():
            k = zip_longest(string2c, string2e)
        else:
            k = zip_longest(string2e, string2c)

        for i, j in k:
            string2_.append(j)
            string2_.append(i)
        string2_ = "".join([i for i in string2_ if i is not None])
        return string2_

    if "$" in strings:
        size = len(strings)
        site_result = re.finditer("\$", strings)
        site = [i.start() for i in site_result]
        assert len(site) >= 2, "$必须成对出现！"

        no_site = [0, ]
        no_site.extend(site)
        no_site.append(size)
        site = np.array(site).reshape((-1, 2))
        string1 = [strings[i[0]:i[1] + 1] for i in site]
        no_site = np.array(no_site).reshape((-1, 2))

        no_site[:, 0] += 1
        no_site[0, 0] = 0
        no_site[-1, -1] = size
        string2 = [strings[i[0]:i[1]] for i in no_site]

        string2 = [subs_en(i) for i in string2 if i not in ['', "$"]]
        string_ = []

        if strings[0] == "$":
            k = zip_longest(string2, string1)
        else:
            k = zip_longest(string1, string2)

        for i, j in k:
            string_.append(j)
            string_.append(i)
        string_ = "".join([i for i in string_ if i is not None])

    else:
        string_ = subs_en(strings)

    string_ = string_.replace(r'$\mathrm{ \; }$', " ")

    return string_


class _BasePlot(object):
    """
    基础画图。

    >>> bp = BasePlot()
    >>> plt = bp.scatter_45_line(x, y_predict, strx='x', stry='y_predict')
    >>> plt.show()
    >>> plt = bp.line_scatter(y_scatter, y_lines, strx='x', stry='ys')
    >>> plt.show()
    >>> plt = bp.lines(ys)
    >>> plt.show()
    >>> plt = bp.lines(ys,x)
    >>> plt.show()

    """

    @staticmethod
    def base_axes():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        return ax, plt

    @staticmethod
    def base_figure():
        plt.figure(0)
        return plt

    @staticmethod
    def scatter(x, y, strx='x', stry='y_predict', line_45=False, color="orange"):
        """
        散点图
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y, marker='o', s=150, alpha=0.7, c=color, linewidths=None, edgecolors='blue')
        if line_45:
            ax.plot([min(x), max(x)], [min(x), max(x)], '--', ms=5, lw=2, alpha=0.7, color='black')
        plt.xlabel(strx)
        plt.ylabel(stry)
        return plt

    def roc_curve(self, y_true, y_predict):
        """ROC 曲线。"""
        fpr, tpr, thresholds = roc_curve(y_true, y_predict)
        roc_auc = auc(fpr, tpr)
        return self.roc(fpr, tpr, roc_auc)

    def roc(self, fpr, tpr, roc_auc):
        """ROC 曲线."""
        # right_index = (tpr + (1 - fpr) - 1)
        # yuzhi = max(right_index)
        # index = right_index.index(max(right_index))
        # tpr_val = tpr(index)
        # fpr_val = fpr(index)

        plt.subplots()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        return plt

    def scatter_45_line(self, y_true, y_predict, strx='x', stry='y_predict', line_45=True, color="cyan"):
        """
        散点图45度线
        """
        return self.scatter(x=y_true, y=y_predict, strx=strx, stry=stry, line_45=line_45, color=color)

    @staticmethod
    def lines(ys, x=None, ys_labels=None, strx='x', stry='ys', no_marker=False, mark_size_ratio=1.0):
        """
        点线图，线图。

        Args:
            ys: (np.ndarray,with shape (n_sample,n_target)), lines
            x:  (np.ndarray),defualt is rank number.
            ys_labels: lables of y.
            strx: (str) axis name
            stry: (str) axis name
            no_marker: (bool), 删除形状
            mark_size_ratio: (float),形状大小。

        """

        markers_ = markers
        if no_marker:
            markers_ = [None] * 100

        if isinstance(ys, pd.DataFrame):
            ys_labels = list(ys.columns.values)
            ys = ys.values
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ys = np.array(ys)
        if not ys_labels or len(ys_labels) != ys.shape[1]:
            labels = ["y%s" % i for i in range(ys.shape[1])]
        else:
            labels = ys_labels
        if x is None:
            for i in range(ys.shape[1]):
                ax.plot(ys[:, i], lw=2, ms=10 * mark_size_ratio, marker=markers_[i], label=labels[i])
        else:
            x = np.array(x)
            for i in range(ys.shape[1]):
                ax.plot(x, ys[:, i], lw=2, ms=10 * mark_size_ratio, marker=markers_[i], label=labels[i])

        plt.xlabel(strx)
        plt.ylabel(stry)
        ax.legend()
        return plt

    @staticmethod
    def line_scatter(x, y_scatter, y_lines, strx='x', stry='ys'):
        """散点，线图独立，合并画图，一般不用"""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y_scatter, marker='o', s=70, alpha=0.7, c='blue', linewidths=None, edgecolors='blue')
        ax.plot(y_lines, '-', lw=2, alpha=0.7, color='black')
        plt.xlabel(strx)
        plt.ylabel(stry)
        return plt

    @staticmethod
    def bar(data, types=None, labels=None, strx='x', stry='ys'):
        """
        柱状图, data为 pandas.Fataframe.
        """

        if isinstance(data, pd.DataFrame):
            types = list(data.columns.values)
            labels = list(data.index.values)
            data = data.values

        data = np.array(data)
        if data.ndim == 1:
            data.reshape((-1, 1))

        index = np.arange(data.shape[0]) * data.shape[1] // 2
        bar_width = 0.35

        if not labels:
            labels = ["f%s" % i for i in index]
        if not types:
            types = list(range(data.shape[1]))
        opacity = 0.4

        fig, ax = plt.subplots()
        for i, (x, typei) in enumerate(zip(data.T, types)):
            ax.bar(index + i * bar_width, x, bar_width, label=typei, alpha=opacity)

        plt.xlabel(strx)
        plt.ylabel(stry)
        ax.set_xticks(index + len(types) * bar_width / 2 - bar_width / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        # fig.tight_layout()

        sns.boxplot()
        return plt

    @staticmethod
    def corr_heatmap(data, square=True, linewidths=.5, annot=False):
        """相关系数图, data使用pandas.DataFrame数据，,strx,stry 必须是列名。"""
        fig = plt.figure()
        fig.add_subplot(111)
        # plt.xticks(rotation='90')
        sns.heatmap(data, cmap="seismic", square=square, linewidths=linewidths, annot=annot,
                    xticklabels=True,
                    yticklabels=True)
        return plt

    @staticmethod
    def violin(strx, stry, data):
        """小提琴图, data为 pandas.Fataframe,strx,stry 必须是列名。"""
        fig = plt.figure()
        fig.add_subplot(111)
        sns.violinplot(x=strx, y=stry, data=data,
                       linewidth=2,  # 线宽
                       width=0.8,  # 箱之间的间隔比例
                       palette='hls',  # 设置调色板
                       order=None,  # 筛选类别
                       scale='area',  # 测度小提琴图的宽度：area-面积相同，count-按照样本数量决定宽度，width-宽度一样
                       gridsize=50,  # 设置小提琴图边线的平滑度，越高越平滑
                       # bw = 0.8        # 控制拟合程度，一般可以不设置
                       hue='smoker',  # 分类
                       split=True,  # 设置是否拆分小提琴图
                       inner="quartile"  # 设置内部显示类型 → “box”, “quartile”, “point”, “stick”, None
                       )
        return plt

    @staticmethod
    def box(strx, stry, data):
        """箱线图，data为 pandas.Fataframe，,strx,stry 必须是列名。"""
        sns.boxplot(x=strx, y=stry, data=data,
                    linewidth=2,  # 线宽
                    width=0.8,  # 箱之间的间隔比例
                    fliersize=3,  # 异常点大小
                    palette='hls',  # 设置调色板
                    whis=1.5,  # 设置IQR
                    notch=True,  # 设置是否以中值做凹槽
                    order=['Thur', 'Fri', 'Sat', 'Sun'],  # 筛选类别
                    )

        sns.swarmplot(x=strx, y=stry, data=data, color='k', size=3, alpha=0.8)
        return plt

    @staticmethod
    def yy_jointplot(strx, stry, data):
        """单/双变量画图，data为 pandas.Fataframe，,strx,stry 必须是列名。"""
        sns.jointplot(strx, stry, data,
                      )
        return plt

    @staticmethod
    def imshow(np_array):
        """热点图"""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np_array)
        return plt

    def show(self):
        plt.show()

    @staticmethod
    def corr_plot(x_cof, x_name=None, left_down="circle", right_top="pie", threshold_left=0.0, threshold_right=0.0,
                  title=None, label_axis="off", front_raito=1., linewidth_ratio=1.):
        """
        相关系数图,比较耗时，不要超过25个特征。

        Examples
        -----------
        >>> data = fetch_california_housing(return_X_y=False)
        >>> name0 = data["feature_names"]
        >>> x = data["data"]
        >>> ys = data["target"]
        >>> x_cof = np.corrcoef(x.T)
        >>> #plot
        >>> plt = BasePlot().corr_plot(x_cof, name0, left_down="circle", right_top="text", threshold_right=0.7, label_axis="off")
        >>> plt.show()

        Parameters
        ----------
        linewidth_ratio:float
            ratio to linewidth.
        x_cof:np.ndarray
            correlation coefficient matrix
        x_name:list,None
            feature names
        left_down:None,"pie","fill","text","circle"
            type for left_down
        right_top:None,"pie","fill","text","circle"
            type for right_top
        threshold_left:float
            threshold for show.
        threshold_right:float
            threshold for show.
        title:str,None
            picture title
        label_axis:"left","right","off"
            label_axis
        front_raito:float
            front scare for show
        """
        assert x_cof.shape[0] == x_cof.shape[1]

        lr = linewidth_ratio

        x_cof = np.round(x_cof, 2)
        if x_name is None:
            x_name = ["$x_{{{i}}}$".format(i=i) for i in range(x_cof.shape[1])]

        size = x_cof
        or_size = np.nan_to_num((abs(size) / size) * (1 - abs(size)))

        n = size.shape[0]
        explode = (0, 0)
        gs = gridspec.GridSpec(n, n)
        gs.update(wspace=0, hspace=0)

        cmap = plt.get_cmap("bwr")  # args
        fill_colors = cmap(size / 2 + 0.5)  # args

        fig = plt.figure(figsize=(12, 12), frameon=True)  # args

        title_fontsize = round(22 * front_raito)  # c_args
        ax_fontsize = round(22 * front_raito)
        score_fontsize = round(16 * front_raito)
        circle_size = round(600 * front_raito)

        fig.text(0.5, 0.05, title, fontsize=title_fontsize, horizontalalignment='center',
                 verticalalignment='center')  # zou, xia

        for i, j in product(range(n), range(n)):
            if j < i and abs(size[i, j]) >= threshold_left:
                types = left_down
            elif j > i and abs(size[i, j]) >= threshold_right:
                types = right_top
            else:
                types = None

            if types == "pie":
                ax = plt.subplot(gs[i, j])
                ax.pie((abs(size[i, j]), abs(or_size[i, j])), explode=explode, labels=None, autopct=None, shadow=False,
                       startangle=90, normalize=False,
                       colors=[fill_colors[i, j], 'w'], wedgeprops=dict(width=1, edgecolor='black', linewidth=0.5),
                       counterclock=False,
                       frame=False, center=(0, 0), )
                ax.set_xlim(-1, 1)
                ax.axis('equal')

            elif types == "fill":
                ax = plt.subplot(gs[i, j])
                ax.set_facecolor(fill_colors[i, j])
                [ax.spines[_].set_color('w') for _ in ['right', 'top', 'left', 'bottom']]

                ax.set_xticks([])
                ax.set_yticks([])

            elif types == "fillandtext":
                ax = plt.subplot(gs[i, j])
                ax.set_facecolor(fill_colors[i, j])
                [ax.spines[_].set_color('w') for _ in ['right', 'top', 'left', 'bottom']]

                ax.text(0.5, 0.5, size[i, j],
                        fontdict={"color": "black"},  # args
                        fontsize=score_fontsize,  # c_arg
                        horizontalalignment='center', verticalalignment='center')
            elif types == "text":
                ax = plt.subplot(gs[i, j])
                ax.text(0.5, 0.5, size[i, j],
                        fontdict={"color": "b"},  # args
                        fontsize=score_fontsize,  # c_arg
                        horizontalalignment='center', verticalalignment='center')
                ax.set_xticks([])
                ax.set_yticks([])
                [ax.spines[_].set_linewidth(rcParams['axes.linewidth'] * lr) for _ in
                 ['right', 'top', 'left', 'bottom']]
                # plt.axis('off')
            elif types == "circle":
                ax = plt.subplot(gs[i, j])
                ax.axis('equal')
                ax.set_xlim(-1, 1)
                ax.scatter(0, 0, color=fill_colors[i, j], s=circle_size * abs(size[i, j]) ** 2)
                ax.set_xticks([])
                ax.set_yticks([])
                [ax.spines[_].set_linewidth(rcParams['axes.linewidth'] * lr) for _ in
                 ['right', 'top', 'left', 'bottom']]
                # plt.axis('off')

            else:
                pass

        for k in range(n):
            ax = plt.subplot(gs[k, k])

            # ax.axis('equal')
            # ax.set_xlim(-1, 1)
            # ax.scatter(0, 0, color=fill_colors[k, k], s=circle_size * abs(size[k, k]))
            # ax.set_xticks([])
            #
            # ax.set_yticks([])

            ax.text(0.5, 0.5, x_name[k], fontsize=ax_fontsize, horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([])
            ax.set_yticks([])
            if label_axis == "left":
                color = ["w", "w", "b", "b"]
                [ax.spines[i].set_color(j) for i, j in zip(['right', 'top', 'left', 'bottom'], color)]
                [ax.spines[_].set_linewidth(rcParams['axes.linewidth'] * lr) for _ in
                 ['right', 'top', 'left', 'bottom']]
            elif label_axis == "right":
                color = ["b", "b", "w", "w"]
                [ax.spines[i].set_color(j) for i, j in zip(['right', 'top', 'left', 'bottom'], color)]
                [ax.spines[_].set_linewidth(rcParams['axes.linewidth'] * lr) for _ in
                 ['right', 'top', 'left', 'bottom']]
            else:
                plt.axis('off')

        @pyplot.FuncFormatter
        def fake_(x, pos):
            return round(2 * (x - 0.5), 1)

        fig.subplots_adjust(right=0.80)
        cbar_ax = fig.add_axes([0.85, 0.125, 0.03, 0.75])
        ColorbarBase(cbar_ax, cmap=cmap, ticks=[0, 0.25, 0.5, 0.75, 1], format=fake_)
        for spine in cbar_ax.spines.values():
            spine.set_linewidth(rcParams['axes.linewidth'] * lr)
        fig.set_size_inches(9, 8.5, forward=True)
        return plt

    @staticmethod
    def _displacement(binary_distance, print_noise=0.001):
        rd = check_random_state(0)
        q = rd.random_sample(binary_distance.shape) * print_noise / 10
        binary_distance = binary_distance + q

        indexs = np.argwhere(binary_distance <= 0)
        indexs = indexs[np.where(indexs[:, 0] > indexs[:, 1])]
        t = rd.random_sample(indexs.shape[0]) * print_noise / 20
        binary_distance[indexs[:, 0], indexs[:, 1]] = t
        binary_distance[indexs[:, 1], indexs[:, 0]] = t
        return binary_distance

    def distance(self, node_color, edge_color_pen=0.7, binary_distance=None, print_noise=0.001,
                 node_name=None, highlight=None):
        """
        kamada_kawai_lay.

        Parameters
        ----------
        highlight:list
            change shape
        slices: list
            the lists of the index of feature subsets, each feature subset is a node.
            Examples 3 nodes
            [[1,4,5],[1,4,6],[1,2,7]]
        node_color: np.ndarray 1D, list, the same size as slices
            the label to classify the node
        edge_color_pen: int
            the transparency of edge between node
        binary_distance: np.ndarray
            distance matrix for each pair node
        print_noise: int
            add noise for less printing overlap
        node_name: list
            x_name of node
        """
        plt.figure()
        g = nx.Graph()

        def _my_ravel(data_cof):
            for i in range(data_cof.shape[0]):
                for k in range(i + 1, data_cof.shape[0]):
                    yield i, k, data_cof[i, k]

        distances = self._displacement(binary_distance, print_noise=print_noise)

        distance_weight = list(_my_ravel(distances))
        g.add_weighted_edges_from(distance_weight)
        # edges=nx.get_edge_attributes(g, 'weight').items()
        edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())
        weights = weights / max(weights)
        pos = nx.layout.kamada_kawai_layout(g)  # calculate site

        if node_name is None:
            le = binary_distance.shape[0]
            lab = {i: i for i in range(le)}

        elif node_name is False:
            lab = None

        else:
            assert binary_distance.shape[0]
            if isinstance(node_name, list) and isinstance(node_name[0], list):
                strr = ","
                node_name = [strr.join(i) for i in node_name]
            lab = {i: j for i, j in enumerate(node_name)}

        nodesize = [600] * distances.shape[0]
        node_edge_color = ["w"] * distances.shape[0]
        if highlight:
            for i in highlight:
                node_edge_color[i] = "aqua"
            for i in highlight:
                nodesize[i] *= 1.3

        mp = {-1: 'gray', 0: "mediumpurple", 1: 'seagreen', 2: 'goldenrod', 3: 'deeppink', 4: "chocolate",
              5: "lightseagreen", }
        node_color = list(map(lambda x: mp[x], node_color))

        nx.draw(g, pos, edgelist=edges, edge_color=np.around(weights, decimals=3) ** edge_color_pen,
                edge_cmap=plt.cm.Blues_r, edge_vmax=0.7,
                width=weights,
                labels=lab, font_size=12,
                node_color=np.array(node_color), vmin=-1,
                edgecolors=node_edge_color, linewidths=1,
                node_size=nodesize,
                )

        return plt


class QPlot(_BasePlot):
    """
    Quick temporary display.

    >>> bp = QPlot()
    >>> plt = bp.scatter_45_line(x, y_predict, strx='x', stry='y_predict')
    >>> plt.show()
    >>> plt = bp.line_scatter(y_scatter, y_lines, strx='x', stry='ys')
    >>> plt.show()
    >>> plt = bp.lines(ys)
    >>> plt.show()
    >>> plt = bp.lines(ys,x)
    >>> plt.show()

    """


class BasePlot(_BasePlot):
    """
    基础画图。

    >>> bp = BasePlot()
    >>> plt = bp.scatter_45_line(x, y_predict, strx='x', stry='y_predict')
    >>> plt.show()
    >>> plt = bp.line_scatter(y_scatter, y_lines, strx='x', stry='ys')
    >>> plt.show()
    >>> plt = bp.lines(ys)
    >>> plt.show()
    >>> plt = bp.lines(ys,x)
    >>> plt.show()

    """

    def __init__(self, font="Arial", figure_size=(10, 7.5), types="single"):
        """

        Args:
            font: (str), 字体。如：【Arial,DejaVu Sans,宋体+Times,宋体，Times】
            figure_size:(tuple) 图片大小。
            types:(str) 图片种类: 如：【single,long,part】
        """
        if types == "single":
            setting_rcParams_single(font=font, figure_size=figure_size)
        elif types == "long":
            setting_rcParams_long(font=font, figure_size=figure_size)
        elif types == "part":
            setting_rcParams_part(font=font, figure_size=figure_size)
        else:
            setting_rcParams_single(font=font, figure_size=figure_size)
