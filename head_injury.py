# Step 1: Extract features

# Read raw data
from pandas import read_csv
data = read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/DAAG/head.injury.csv")
print('raw data: \n', data.head())

data = data.drop((['Unnamed: 0']), axis=1)
print('\nprocessed data: \n', data.head())

dataset = data.values

print('\nvalues: \n', dataset[0:5, :])

# Extract features and labels
X = dataset[:, 0:10].astype(float)
Y = dataset[:, 10]

print('\nX: \n', X.shape)
print('\nX: \n', X[0:5, :])
print('\nY: \n', Y.shape)
print('\nY: \n', Y[0:5])

# Normalize or numberize labels 
# This step is skipped if labels are numberical. It remains here for reference only.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print('\nencoded_Y: \n', encoded_Y[0:5])

# Split train set and test set randomly
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.2, random_state=100)

print('\nX_train: ', X_train.shape)
print('\nX_test: ', X_test.shape)
print('\ny_train: ', y_train.shape)
print('\ny_test: ', y_train.shape)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
tX_train = scaler.transform(X_train)
tX_test = scaler.transform(X_test)

dims = tX_train.shape[1]

print('\ntX_train: \n', tX_train[0:5, :])
print('\ntX_test: \n', tX_test[0:5, :])


# Step 2: Build model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=1,
                input_shape=(dims,),
                activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()              

# Step 3: Train 
history = model.fit(tX_train, y_train, batch_size=50, epochs=100, verbose=1)

# illustrate how loss and accuracy
plt.plot(history.history['loss'], '--b')
plt.plot(history.history['accuracy'], '--r')
plt.title('training loss/accuracy')
plt.ylabel('loss/accuracy')
plt.xlabel('epochs')
plt.show()

# Step 4: Test
eval_ret = model.evaluate(tX_test, y_test)
print('\n%s: %.02f' % (model.metrics_names[0], eval_ret[0]))
print('\n%s: %.02f%%' % (model.metrics_names[1], eval_ret[1]*100))
