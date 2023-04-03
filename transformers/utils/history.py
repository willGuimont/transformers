from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class History:
    def __init__(self, logs_list=None):
        """
        Util class to save and display graphs for training metrics.
        Adapted from https://github.com/ulaval-damas/glo4030-labs
        :param logs_list: list of dict containing the different metrics for each epoch.
                          The dict should have the same keys as taken by the save() method.
        """
        self.history = defaultdict(list)

        if logs_list is not None:
            for logs in logs_list:
                self.save(logs)

    def save(self, logs):
        """
        Save metrics in the history.
        :param logs: list of dict containing the different metrics for each epoch.
                     The dict should have the same keys as taken by the save() method.
                     Can also contain the key 'lr' for the learning rate.
        :return:
        """
        for k, v in logs.items():
            self.history[k].append(v)

    def display(self, display_lr=False):
        """
        Display two graphs for the training and validation losses and for the training and validation accuracy.
        Also, if the learning rate has been saved, it is possible to display a graph for it.
        :param display_lr: if we want to display a graph for the learning rate. By default, the graph is not displayed.
        :return:
        """
        epoch = len(self.history['loss'])
        epochs = list(range(1, epoch + 1))

        num_plots = 3 if display_lr else 2
        _, axes = plt.subplots(num_plots, 1, sharex=True)
        plt.tight_layout()

        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs, self.history['acc'], label='Train')
        axes[0].plot(epochs, self.history['val_acc'], label='Validation')
        axes[0].legend()

        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, self.history['loss'], label='Train')
        axes[1].plot(epochs, self.history['val_loss'], label='Validation')

        if display_lr and 'lr' in self.history:
            axes[2].set_xlabel('Epochs')
            axes[2].set_ylabel('Lr')
            axes[2].plot(epochs, self.history['lr'], label='Lr')
            axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            axes[1].set_xlabel('Epochs')
            axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.show()

    def display_loss(self):
        """
        Display a graph for the training and validation losses.
        :return:
        """
        epoch = len(self.history['loss'])
        epochs = list(range(1, epoch + 1))

        plt.tight_layout()

        plt.plot(epochs, self.history['loss'], label='Train')
        plt.plot(epochs, self.history['val_loss'], label='Validation')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()
