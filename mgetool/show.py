#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/29 19:46
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
Notes:
    add definition to matpltlib.plt to draw better figures.
    Plot corr.
"""

# """
#
# eg:
# # fig = plt.figure(figsize=[6.4, 4.8])
# # # fig.patch.set_facecolor("white")
# # ax = fig.add_subplot(111)
# # # # ax = fig.add_axes([0.15,0.1,0.7,0.3])
# # # ax.patch.set_facecolor("w")
# # # # ax.patch.set_alpha(0.5)
# # # [ax.spines[_].set_linewidth(3) for _ in ['left', 'right', 'bottom', 'top']]
# # # ax.tick_params(direction='in', which='major', width=3, length=7, colors='black', labelsize=15)
# # plt.xlabel("x", )
# # plt.ylabel("y", )
# # plt.title("x_name")
# # # plt.xlim()
# # # plt.ylim()
# # # plt.legend()
# # plt.show()
#
# plt.sca(ax1)
# plt.sca(ax2)
# """

from itertools import product

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colorbar import ColorbarBase
from scipy.stats import pearsonr
from sklearn.datasets import load_boston
from sympy.physics.quantum.circuitplot import pyplot


class BasePlot(object):
    """
    Add definition to matpltlib.plt to draw better figures

    >>> bp = BasePlot()
    >>> plt = bp.yy_numpy(y_true, y_predict, strx='y_true', stry='y_predict')
    >>> plt.show()
    >>> plt = bp.line_scatter(y_scatter, y_lines, strx='x', stry='y')
    >>> plt.show()
    >>> plt = bp.lines(y)
    >>> plt.show()
    >>> plt = bp.lines(y,x)
    >>> plt.show()

    """

    def __init__(self, font="Times"):
        if font == "Times":
            rcParams['font.family'] = 'serif'
            rcParams['font.serif'] = ['Times new roman']
        # rcParams['figure.dpi'] = 300
        # rcParams['figure.figsize'] = [6.4, 4.8]
        # rcParams['figure.titlesize'] = 20

        rcParams['axes.titlesize'] = 18
        rcParams['axes.labelsize'] = 16
        rcParams['axes.linewidth'] = 2

        rcParams['xtick.major.width'] = 2
        rcParams['ytick.major.width'] = 2
        rcParams['xtick.major.size'] = 4
        rcParams['ytick.major.size'] = 4

        # rcParams['xtick.labelsize'] = 14
        # rcParams['ytick.labelsize'] = 14
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.direction'] = 'in'

        rcParams['legend.fontsize'] = 14

        rcParams['axes.grid'] = True
        rcParams['grid.alpha'] = 0.1

    @staticmethod
    def base_axes():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        return ax, plt

    @staticmethod
    def base_figure():
        plt.figure(0)
        return plt

    def yy_numpy(self, y_true, y_predict, strx='y_true', stry='y_predict'):
        """
        y_true vs y_predict
        """

        return self.scatter(y_true, y_predict, strx=strx, stry=stry)

    @staticmethod
    def bar(data, types=None, labels=None, strx='x', stry='y'):

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
    def scatter(y_true, y_predict, strx='y_true', stry='y_predict'):
        x, y = y_true, y_predict
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y, marker='o', s=50, alpha=0.7, c='orange', linewidths=None, edgecolors='blue')
        ax.plot([min(x), max(x)], [min(x), max(x)], '--', ms=5, lw=2, alpha=0.7, color='black')
        plt.xlabel(strx)
        plt.ylabel(stry)
        return plt

    @staticmethod
    def lines(y, x=None, line_labels=None, strx='x', stry='y', ):
        if isinstance(y, pd.DataFrame):
            line_labels = list(y.columns.values)
            y = y.values
        fig = plt.figure()
        ax = fig.add_subplot(111)
        y = np.array(y)
        if not line_labels or len(line_labels) != y.shape[1]:
            labels = ["f%s" % i for i in range(y.shape[1])]
        else:
            labels = line_labels
        if x is None:
            if y.ndim == 2:
                for i in range(y.shape[1]):
                    ax.plot(y[:, i], lw=1.5, marker='o', label=labels[i])
            if y.ndim == 1:
                ax.plot(y, lw=1.5, marker='o', label=labels[0])
        else:
            x = np.array(x)
            if y.ndim == 2:
                for i in range(y.shape[1]):
                    ax.plot(x, y[:, i], lw=1.5, marker='o', label=labels[i])
            if y.ndim == 1:
                ax.plot(x, y, lw=1.5, marker='o', label=labels[0])
        plt.xlabel(strx)
        plt.ylabel(stry)
        ax.legend()
        return plt

    @staticmethod
    def line_scatter(x, y_scatter, y_lines, strx='x', stry='y'):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y_scatter, marker='o', s=10, alpha=0.7, c='orange', linewidths=None, edgecolors='blue')
        ax.plot(y_lines, '-', ms=5, lw=2, alpha=0.7, color='black')
        plt.xlabel(strx)
        plt.ylabel(stry)
        return plt

    @staticmethod
    def corr(data, square=True, linewidths=.5, annot=False):
        fig = plt.figure()
        fig.add_subplot(111)
        # plt.xticks(rotation='90')
        sns.heatmap(data, cmap="seismic", square=square, linewidths=linewidths, annot=annot, xticklabels=True,
                    yticklabels=True)
        return plt

    @staticmethod
    def violin(strx, stry, data):
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
    def yy_jointplot(str_x, str_y, data):
        sns.jointplot(str_x, str_y, data,
                      kind='reg')
        return plt

    @staticmethod
    def imshow(np_array):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np_array)
        return plt

    def show(self):
        plt.show()


def lin_cof(x0):
    results_list = []
    xx = x0.T
    yy = x0.T
    for a in xx:
        for b in yy:
            results = pearsonr(a, b)[0]
            results_list.append(results)
    results1 = np.array(results_list).reshape((x0.shape[-1], x0.shape[-1]))
    return results1


def corr_plot(x_cof, x_name=None, left_down=None, right_top=None, threshold_left=0.0, threshold_right=0.9,
              title="pearsonr coefficient", label_axis="off", front_raito=1):
    """
    Plot corr

    Examples
    -----------
    >>> data = load_boston(return_X_y=False)
    >>> name0 = data["feature_names"]
    >>> x = data["data"]
    >>> y = data["target"]
    >>> x_cof = np.corrcoef(x.T)
    >>> #plot
    >>> plt = corr_plot(x_cof, name0, left_down="circle", right_top="text", threshold_right=0.7, label_axis="off")
    >>> plt.show()

    Parameters
    ----------
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
    title:str
        picture title
    label_axis:"left","right","off"
        label_axis
    front_raito:float
        front scare for show
    """
    x_cof = np.round(x_cof, 2)

    name = x_name

    size = x_cof
    or_size = np.nan_to_num((abs(size) / size) * (1 - abs(size)))

    n = size.shape[0]
    explode = (0, 0)
    gs = gridspec.GridSpec(n, n)
    gs.update(wspace=0, hspace=0)

    cmap = plt.get_cmap("bwr")  # args
    fill_colors = cmap(size / 2 + 0.5)  # args

    fig = plt.figure(figsize=(6, 6), frameon=True)  # args

    title_fontsize = round(15 * front_raito)  # c_args
    ax_fontsize = round(14 * front_raito)
    score_fontsize = round(12 * front_raito)
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
            ax.pie((size[i, j], or_size[i, j]), explode=explode, labels=None, autopct=None, shadow=False,
                   startangle=90,
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
            # plt.axis('off')
        elif types == "circle":
            ax = plt.subplot(gs[i, j])
            ax.axis('equal')
            ax.set_xlim(-1, 1)
            ax.scatter(0, 0, color=fill_colors[i, j], s=circle_size * abs(size[i, j]) ** 2)
            ax.set_xticks([])

            ax.set_yticks([])
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

        ax.text(0.5, 0.5, name[k], fontsize=ax_fontsize, horizontalalignment='center', verticalalignment='center')
        ax.set_xticks([])
        ax.set_yticks([])
        if label_axis == "left":
            color = ["w", "w", "b", "b"]
            [ax.spines[i].set_color(j) for i, j in zip(['right', 'top', 'left', 'bottom'], color)]
        elif label_axis == "right":
            color = ["b", "b", "w", "w"]
            [ax.spines[i].set_color(j) for i, j in zip(['right', 'top', 'left', 'bottom'], color)]
        else:
            plt.axis('off')

    @pyplot.FuncFormatter
    def fake_(x, pos):
        return round(2 * (x - 0.5), 1)

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.80, 0.125, 0.03, 0.75])
    ColorbarBase(cbar_ax, cmap=cmap, ticks=[0, 0.25, 0.5, 0.75, 1], format=fake_)
    fig.set_size_inches(7, 6, forward=True)
    return plt


if __name__ == '__main__':
    data = load_boston(return_X_y=False)

    name0 = data["feature_names"]
    x = data["data"]
    y = data["target"]
    x_cof = np.corrcoef(x.T)

    corr_plot(x_cof, name0, left_down="circle", right_top="text", threshold_right=0.7, label_axis="off",
              front_raito=0.6)
    plt.show()
