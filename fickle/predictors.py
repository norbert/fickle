__all__ = ['GenericSVMClassifier']

from sklearn import svm

from fickle.backend import Backend


class GenericSVMClassifier(Backend):

    @staticmethod
    def model():
        return svm.LinearSVC()
