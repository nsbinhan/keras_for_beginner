# Objective: classify 3 species of iris

# Step 1: Extract features

# Read raw data
from pandas import read_csv
data = read_csv("https://www.openml.org/data/get_csv/61/dataset_61_iris.csv")

print('raw data: \n', data.head())

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

# Numberize labels 
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
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print('\nty_train: \n', y_train[0:5])
print('\nty_test: \n', y_test[0:5])

# Step 2: Build model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D

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
history = model.fit(X_train, y_train, batch_size=16, epochs=100, validation_split = 0.2, verbose=0)

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
eval_ret = model.evaluate(X_test, y_test)
print('\n%s: %.02f' % (model.metrics_names[0], eval_ret[0]))
print('\n%s: %.02f%%' % (model.metrics_names[1], eval_ret[1]*100))

# Note: increasing output dimension of Dense layer might enhance accuracy
# Ex: model.add(Dense(64, input_dim=dims, activation='relu'))
