import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.metrics import confusion_matrix
import numpy as np
# The digits dataset
digits = datasets.load_digits()

# plotting confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix',cmap=plt.cm.Blues):
  
    plt.figure(1)
    plt.imshow(cm, cmap = cmap,interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(digits.target_names))
    plt.xticks(tick_marks, digits.target_names, rotation=0)
    plt.yticks(tick_marks,digits.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
   
   


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# Classify using svm
classifier = svm.SVC(kernel='linear', C=1, gamma=0.001)

# Training
classifier.fit(data[:(2*n_samples)//3], digits.target[:(2*n_samples)//3])

# Now predict the value of the digit on the second half:
expected = digits.target[(2*n_samples)//3:]
predicted = classifier.predict(data[(2*n_samples)//3:])

cm = confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

plot_confusion_matrix(cm)

plt.figure(2)
plt.title("Test Image of 0")
plt.imshow(digits.images[0], interpolation='nearest',cmap=plt.cm.gray_r)
print("Prediction - %d" % classifier.predict(data[0:1]))
