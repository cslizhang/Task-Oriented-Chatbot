# -*- coding: utf-8 -*-
# @Time    : 5/13/18 14:24
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import tensorflow as tf


class IntentRNNAttention(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def load(self):
        pass

    def save(self):
        pass

    def predict(self):
        pass

    def predict_prob(self):
        pass

    def train_on_batch(self):
        pass
