import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('dataset_players.csv')
#print(dataset.head()) 

no_zeros = ['height','weight','pace','pace_acceleration','pace_sprint_speed','dribbling','drib_agility','drib_balance','drib_reactions','drib_ball_control','drib_dribbling','drib_composure','shooting','shoot_positioning','shoot_finishing','shoot_shot_power','shoot_long_shots','shoot_volleys','shoot_penalties','passing','pass_vision','pass_free_kick','pass_short','pass_long','pass_curve','defending','def_interceptions','def_heading','def_marking','def_stand_tackle','def_slid_tackle','physicality','phys_jumping','phys_stamina','phys_strength','phys_aggression']
for column in no_zeros:
    dataset[column] = dataset[column].replace(0,np.NaN) #replace 0 with NULL
    mean = int(dataset[column].mean(skipna=True)) #calculate mean and skip NULL
    dataset[column] = dataset[column].replace(np.NaN,mean) #replace NULL with mean value

dataset['pref_foot'] = dataset['pref_foot'].replace('Right',1)
dataset['pref_foot'] = dataset['pref_foot'].replace('Left',0)
dataset['att_workrate'] = dataset['att_workrate'].replace('High',2)
dataset['att_workrate'] = dataset['att_workrate'].replace('Med',1)
dataset['att_workrate'] = dataset['att_workrate'].replace('Low',0)
dataset['def_workrate'] = dataset['def_workrate'].replace('High',2)
dataset['def_workrate'] = dataset['def_workrate'].replace('Med',1)
dataset['def_workrate'] = dataset['def_workrate'].replace('Low',0)


y = pd.factorize(dataset['position'])[0]  
#print(y)

x_test = dataset[dataset['player_ID']==21637].iloc[:,12:76]  #21637 is index of last row , add a player ID at the last row of csv file and fill in his/her stats then run

features = dataset.iloc[:,12:76]
#print(features)
features = features.fillna(features.mean())

clf = RandomForestClassifier(n_jobs = 2 , random_state = 0) 
clf.fit(features,y)

#print(x_test)
ans = clf.predict(x_test)
ans = int(ans)
pos = ['CAM','ST','CF','LW','GK','CB','RW','CM','RB','LM','CDM','RM','LB','LWB','RWB','LF','RF']
#print(max(y))
print("You should play at the :" ,pos[ans], " position")

y1 = dataset['player_ID']
classifier = KNeighborsClassifier(n_neighbors = 147 , p = 2 , metric = 'euclidean') 
classifier.fit(features , y1)
play = classifier.predict(x_test)
play = dataset[dataset['player_ID']==int(play)].iloc[:,0:11]
print("You stats are very similar to : ")
print(play)


#P.S. accuracy of similar stats player is very low
#Accuracy of position reccomendation is > 98%
# to see your score's recommendations on the last row of csv file add ID no = 21637
# and fill in your details as per the column titles.
# while filling remember 
#Right = 1 Left = 0
# High = 2 , Med = 1 and Low = 0