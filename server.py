from fickle import API

from svm_classifier import GenericSVMClassifier
backend = GenericSVMClassifier()

app = API(__name__, backend)
app.run(debug = True)
