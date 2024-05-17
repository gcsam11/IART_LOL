import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn as sk

# Load the data
data = pd.read_csv('match_data_v5.csv')

data.columns =["matchID","blueTeamControlWardsPlaced","blueTeamWardsPlaced","blueTeamTotalKills","blueTeamDragonKills","blueTeamHeraldKills","blueTeamTowersDestroyed","blueTeamInhibitorsDestroyed","blueTeamTurretPlatesDestroyed","blueTeamFirstBlood","blueTeamMinionsKilled","blueTeamJungleMinions","blueTeamTotalGold","blueTeamXp","blueTeamTotalDamageToChamps","redTeamControlWardsPlaced","redTeamWardsPlaced","redTeamTotalKills","redTeamDragonKills","redTeamHeraldKills","redTeamTowersDestroyed","redTeamInhibitorsDestroyed","redTeamTurretPlatesDestroyed","redTeamMinionsKilled","redTeamJungleMinions","redTeamTotalGold","redTeamXp","redTeamTotalDamageToChamps","blueWin","empty"]
data = data.drop(columns=['matchID'])
data = data.drop(columns=['empty'])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

all_inputs = data.drop('blueWin', axis=1)
all_labels = data['blueWin'].values

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_inputs, all_labels, test_size=0.10, random_state=1)

print(data["blueWin"].value_counts())
print(data["blueWin"].value_counts(normalize=True))

# Train the model decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(
    max_depth=3

)
'''
model.fit(training_inputs, training_classes)

print("Decision Tree")
print(model.score(testing_inputs, testing_classes))

# Train the model SVM
from sklearn.svm import SVC
model = SVC()
model.fit(training_inputs, training_classes)

print("SVM")
print(model.score(testing_inputs, testing_classes))

# Train the model KNN

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

model.fit(training_inputs, training_classes)

print("KNN")
print(model.score(testing_inputs, testing_classes))

'''

from sklearn.model_selection import cross_val_score

decision_tree_classifier = DecisionTreeClassifier()

# cross_val_score returns a list of the scores, which we can visualize
# to get a reasonable estimate of our classifier's performance
cv_scores = cross_val_score(decision_tree_classifier, all_inputs, all_labels, cv=10)
plt.hist(cv_scores)
plt.title('Average score: {}'.format(np.mean(cv_scores)))

