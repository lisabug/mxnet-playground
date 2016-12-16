#!/usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import re
from itertools import cycle

colors = cycle(["aqua", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon", "navy", "olive", "purple", "red", "silver", "teal", "yellow"])

def read_log(log):
    with open(log) as f:
        content = f.readlines()
    return content


def parse_log(re_exp, log):
    score = []
    for line in log:
        s = re.search(re_exp, line)
        if s:
            score.append(float(s.groups()[0]))
    return score


def plot_log(re_exp, **kwargs):
    results = {}
    for log_file in kwargs['logs']:
        log_content = read_log(log_file)
        log_name = log_file.split('/')[-1].split('.')[0]
        score = parse_log(re_exp, log_content)
        results[log_name] = score
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for n, s in results.items():
        ax.plot(range(len(s)), s, label=n, color=next(colors))

    ax.legend(loc='best')
    ax.margins(0.01)
    ax.set_yscale('linear')

    ax.set_xlabel(kwargs['xlabel'])
    ax.set_ylabel(kwargs['ylabel'])
    ax.set_title(kwargs['title'])
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    re_exp = '.*Epoch\[\d+\]\sValidation-accuracy=(.+)'
    logs = ['../../layer-normalization/log/ln_bs4.log', '../../layer-normalization/log/no_ln_bs4.log']
    plot_log(re_exp,
             **{'logs': logs,
                'xlabel': 'Epoch',
                'ylabel': 'Accuracy',
                'title' : 'Title'})

