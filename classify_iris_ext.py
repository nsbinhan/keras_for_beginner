# Objective: classify 3 species of iris
# Note: Support early stopping and demonstrate kappa and confusion matrix

# Step 1: Extract features

# Read raw data
from pandas import read_csv
data = read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv")

print('raw data: \n', data.head())

data = data.drop((['Unnamed: 0']), axis=1)
print('\nprocessed data: \n', data.head())

dataset = data.values

print('\nvalues: \n', dataset[0:5, :])

# Extract features and labels
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]
num_classes = 3

print('\nX: \n', X.shape)
print('\nX: \n', X[0:5, :])
print('\nY: \n', Y.shape)
print('\nY: \n', Y[0:5])

# Normalize or numberize labels 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print('\nencoded_Y: \n', encoded_Y[0:5])

# Split train set and test set randomly
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.25, random_state=45)

print('\nX_train: ', X_train.shape)
print('\nX_test: ', X_test.shape)
print('\ny_train: ', y_train.shape)
print('\ny_test: ', y_train.shape)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
tX_train = scaler.transform(X_train)
tX_test = scaler.transform(X_test)

# For softmax, X_ values must be in range of [0, 1], y_ is one-hot-coding
# idx_train = X_train.argmax(0)
# max_train = [max(X_train[:, i]) for i in range(X_train.shape[1])]
# print('idx_train ', idx_train)
# print('max_train ', max_train)

# for i in range(len(max_train)):
#     if(max_train[i] != 0):
#         X_train[:, i] /= max_train[i]
    
# idx_test = X_test.argmax(0)
# max_test = [max(X_test[:, i]) for i in range(X_test.shape[1])]
# for i in range(len(max_test)):
#     if(max_test[i] != 0):
#         X_test[:, i] /= max_test[i]
    
dims = X_train.shape[1]

print('\ntX_train: \n', X_train[0:5, :])
print('\ntX_test: \n', X_test[0:5, :])

# backup 
by_test = y_test
from keras.utils import to_categorical
cy_train = to_categorical(y_train, num_classes)
cy_test = to_categorical(y_test, num_classes)

print('\nty_train: \n', cy_train[0:5])
print('\nty_test: \n', cy_test[0:5])

# Step 2: Build model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

cb_early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

model = Sequential()
model.add(Dense(32, input_dim=dims, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['categorical_accuracy'])

model.summary()      

# Step 3: Train 
history = model.fit(X_train, 
                    cy_train,
                    batch_size=16, 
                    epochs=100, 
                    validation_split = 0.2, 
                    verbose=1,
                    callbacks=[cb_early_stop])

# illustrate how loss and accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], '--b')
plt.plot(history.history['categorical_accuracy'], '--r')
plt.title('loss/accuracy')
plt.ylabel('loss/accuracy')
plt.xlabel('epochs')
plt.legend(['loss', 'accuracy'], loc='upper right')
plt.show()

plt.plot(history.history['loss'], '--b')
plt.plot(history.history['val_loss'], '--r')
plt.title('loss function')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Step 4: Test
eval_ret = model.evaluate(X_test, cy_test)
print('\n%s: %.02f' % (model.metrics_names[0], eval_ret[0]))
print('\n%s: %.02f%%' % (model.metrics_names[1], eval_ret[1]*100))

y_pred = model.predict_classes(tX_test)

# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# Review
from sklearn.metrics import confusion_matrix
import numpy as np

class_names = np.unique(Y)
cf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cf_matrix, class_names)

from sklearn.metrics import cohen_kappa_score as kappa
kappa(y_test, y_pred)    
