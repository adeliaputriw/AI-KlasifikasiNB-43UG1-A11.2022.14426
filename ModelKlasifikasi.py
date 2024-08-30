import pandas as pd
penguin = pd.read_csv('penguin.csv')
data = penguin.copy()
target = 'species'
encode = ['island' , 'sex']
for col in encode:
    dummy = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data,dummy], axis=1)
    del data[col]
    
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]
data['species'] = data['species'].apply(target_encode)

## Proses Learning ##
X = data.drop('species', axis=1)
Y = data['species']

## Model Klasifikasi Naive Bayes ##
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, Y)

## Menyimpan Model NBC ##
import pickle
pickle.dump(model, open('modelNBC_penguin.pkl', 'wb'))
