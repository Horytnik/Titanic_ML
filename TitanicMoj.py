#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv(r"train.csv")
train_data.head()

test_data = pd.read_csv(r"test.csv")
test_data.head() 

with pd.option_context('display.max_columns', None):  # more options can be specified also
    print(test_data)

print(test_data.info())

women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']['Survived']
classmen = train_data.loc[train_data.Sex == 'male']['Pclass']
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)


classmencombined = pd.concat([men, classmen], axis=1, sort=False)
firstclassmen = classmencombined.loc[classmencombined.Pclass == 1]['Survived']
secondclassmen = classmencombined.loc[classmencombined.Pclass == 2]['Survived']
thirdclassmen = classmencombined.loc[classmencombined.Pclass == 3]['Survived']

survived_1_class_men = sum(firstclassmen)/len(firstclassmen)
survived_2_class_men = sum(secondclassmen)/len(secondclassmen)
survived_3_class_men = sum(thirdclassmen)/len(thirdclassmen)

print("% of 1st class men: who survived",survived_1_class_men)
print("% of 2nd class men: who survived",survived_2_class_men)
print("% of 3rd class men: who survived",survived_3_class_men)


peop1stclass = train_data.loc[train_data.Pclass == 1]['Survived']
peop2ndclass = train_data.loc[train_data.Pclass == 2]['Survived']
peop3rdclass = train_data.loc[train_data.Pclass == 3]['Survived']

print("$ of people from 1st class who survived", sum(peop1stclass)/len(peop1stclass))
print("$ of people from 2nd class who survived", sum(peop2ndclass)/len(peop2ndclass))
print("$ of people from 3rd class who survived", sum(peop3rdclass)/len(peop3rdclass))

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

