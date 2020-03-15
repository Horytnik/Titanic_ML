import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn



train_data = pd.read_csv(r"train.csv")
test_data = pd.read_csv(r"test.csv")


with pd.option_context('display.max_columns', None):  # more options can be specified also
    print(test_data)

women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']['Survived']
classmen = train_data.loc[train_data.Sex == 'male']['Pclass']
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

labels = ["Woman", "Men"]

graph = plt.subplot()
bar1 = graph.bar(0.5, rate_women *100, 0.2)
bar2 = graph.bar(1, rate_men * 100, 0.2)

graph.set_ylabel("%")
graph.set_title('Percantage of Woman and Man who survived')
graph.set_xticks([0.5, 1])
graph.set_xticklabels(labels)

plt.show()

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

labels = ['1st class','2nd class','3rd class']
x = np.arange(len(labels))
graph = plt.subplot()
bar1 = graph.bar (0.5, sum(peop1stclass)/len(peop1stclass)*100, 0.5, label = "1st class")
bar2 = graph.bar (1.5, sum(peop2ndclass)/len(peop2ndclass)*100, 0.5, label = '2nd class')
bar3 = graph.bar (2.5, sum(peop3rdclass)/len(peop3rdclass)*100, 0.5, label = '3rd class')

graph.set_ylabel('%')
graph.set_title('Percantage of people who survived based on class')
graph.set_xticks(x+0.5)
graph.set_xticklabels(labels)

plt.show()


graphSurAge = seaborn.FacetGrid(train_data, col="Survived")
graphSurAge.map(plt.hist,'Age', bins =30)
plt.show()

graphClasSur = seaborn.FacetGrid(train_data, col= "Survived", row="Pclass")
graphClasSur.map(plt.hist, 'Age', bins = 20)
plt.show()

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

modelRandomForest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
modelRandomForest.fit(X, y)
predictions = modelRandomForest.predict(X_test)

print(modelRandomForest.score(X,y))

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('RandomForestSubmission.csv', index=False)


modelDecisionTree = DecisionTreeClassifier()
modelDecisionTree.fit(X,y)
predictions = modelDecisionTree.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('DecisionTreeSubmission.csv', index=False)

