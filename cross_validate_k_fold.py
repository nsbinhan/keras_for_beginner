# Objective: Use logistic regression model to match roughly the clinical characteristics of a 
# sample of individuals who suffered minor head injuries

# Step 1: Extract features

# Read raw data
from pandas import read_csv
# data = read_csv("https://www.openml.org/data/get_csv/54003/MagicTelescope.csv")
data = read_csv("https://www.openml.org/data/get_csv/37/dataset_37_diabetes.csv")
print('raw data: \n', data.head())

dataset = data.values

# Extract features and labels
X = dataset[:, 0:8].astype(float)
Y = dataset[:, 8]

# Normalize or numberize labels 
# This step is skipped if labels are numberical. It remains here for reference only.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

test_debug = True

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
tX = scaler.transform(X)

dims = X.shape[1]

# Step 2: Build model
from keras.models import Sequential
from keras.layers import Dense, Dropout

# function to create model, used in KerasClassifier
def create_model():
    model = Sequential()
    model.add(Dense(32, input_dim=dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.summary()              
    return model
  
# Step 3: Validate
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

seed = 100
np.random.seed(seed)

# cross-validation object
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# kfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=seed)

# Solution 1: KerasClassifier
from keras.wrappers.scikit_learn import KerasClassifier
# create classification model
kc_model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=50)
# Evaluate model by cross-validation
results = cross_val_score(kc_model, tX, encoded_Y, cv=kfold)
print('KerasClassifier: %.2f (+/- %.2f)' % (np.mean(csvscores), np.std(csvscores)))

# Solution 2: Manual
model = create_model()
csvscores = []
for train, test in kfold.split(tX, encoded_Y):
    # fit the model
    model.fit(tX[train], encoded_Y[train], epochs=100, batch_size=50)
    # evaluate the model
    scores = model.evaluate(tX[test], encoded_Y[test])
    print('%s: %.2f' % (model.metrics_names[1], scores[1]))
    csvscores.append(scores[1])
    
print('Manual: %.2f (+/- %.2f)' % (np.mean(csvscores), np.std(csvscores)))
