from fickle import Backend

from sklearn import svm

class GenericSVMClassifier(Backend):
    @staticmethod
    def model():
        return svm.LinearSVC()
