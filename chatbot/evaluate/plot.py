# coding:utf8
# @Time    : 18-5-21 下午3:03
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc')


def plot_confusion_matrix(y_true, y_test, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_test)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

import seaborn as sns

def plot_attention_1d(values, attentions, **kwargs):
    if attentions.ndim == 1:
        attentions = attentions.reshape(1, -1)
        values = [values]
    fig, ax = plt.subplots(**kwargs)
    im = ax.imshow(attentions, interpolation='nearest',
                   vmin=attentions.min(), vmax=attentions.max())
    plt.colorbar(im, shrink=0.5, ticks=[0,1], cmap=plt.cm.summer)

    # We want to show all ticks...
    # ax.set_xticks(np.arange(len(attentions)))
    # ax.set_yticks(np.arange(len(attentions)))
    # ... and label them with the respective list entries
    # ax.set_xticklabels(farmers)
    # ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(attentions.shape[0]):
        for j in range(attentions.shape[1]):
            text = ax.text(j, i, values[i][j],
                           ha="center", va="center", color="w", size=10, fontproperties=zhfont)

    ax.set_title("Atention 可视化", fontproperties=zhfont)
    fig.tight_layout()
    plt.show()

# plot_attention_1d([str(i) for i in range(10)], np.abs(np.random.rand(10)), )

