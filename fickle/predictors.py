__all__ = ['GenericSVMClassifier']

from sklearn.svm import LinearSVC

from .backend import Backend


class GenericSVMClassifier(Backend):

    @staticmethod
    def model():
        return LinearSVC()
