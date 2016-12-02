###########################################
# Robert Dickerson, John Krieger, Logan Skidmore the cool one
# Brother Burton
# CS 450 Machine Learning Project
###################################################
import dota2api
import random
import numpy as np
import sys
import pandas
from sklearn import datasets
from sklearn.cross_validation import train_test_split as tsp
import csv



def getKAmount():
    k = 0
    while k < 1:
        k = int(input("Please enter the nearest neighbor you want the program to search to: "))

    return k



class KNN:

    def predict(self, train_data, train_target, test_data, k):


        nInputs = np.shape(test_data)[0]
        closest = np.zeros(nInputs)

        for n in range(nInputs):

            # Compute distances
            distances = np.sum((train_data-test_data[n,:])**2, axis=1)

            indices = np.argsort(distances,axis=0)

            classes = np.unique(train_target[indices[:k]])
            if len(classes)==1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes)+1)
                for i in range(k):
                    counts[train_target[indices[i]]] += 1
                closest[n] = np.max(counts)

        return closest



    def train(self, data_set, target_set):
        self.trainingData = np.asarray(data_set)
        self.testingData = np.asarray(target_set)
        print("\nThe system has been trained on the new set of data.\n")


def get_accuracy(results_of_predict, test_targets):
    value_correct = 0
    for i in range(test_targets.size):
        value_correct += results_of_predict[i] == test_targets[i]



def train_system(data, target, classifier):
    #random.shuffle(iris.data)
    testAmount = float(0.3)
    timesShuffled = 15
    k = getKAmount()

    train_data, test_data, train_target, test_target = tsp(data, target, test_size = testAmount,
                                                           random_state = timesShuffled)

    classifier.train(train_data, train_target)
    get_accuracy(classifier.predict(train_data, train_target, test_data, k), test_target)

def main(argv):
    api = dota2api.Initialise("7FCC616D07990B76386BCE8AB2F51B32")
    matches = api.get_match_history_by_seq_num(start_at_match_seq_num=2300000000)
    print matches['status']
    print(len(matches['matches']))
    for mat in matches['matches']:
        print mat['match_id']
    match = api.get_match_details(match_id=2813794044)
    print ("Match win:")
    # 'radiant_win' says if the radiant team won. if false dire team won. (whatever that means)
    print (match['radiant_win']),(match['duration'])
    players = (match['players'])

    rad_zeroes = [0] * 112
    dire_zeroes = [0] * 112
    gameResults = []

    player = players[0]
    print player['hero_id']
    print player['hero_name']
    heroesList = api.get_heroes()
    print heroesList['heroes']

    if (match['radiant_win']):
        radiant_team ='Win'
        dire_team = 'Loss'
    else:
        radiant_team = 'Loss'
        dire_team ='Win'

    # these are the match players
    for i in range(0,5):
        #print "Radiant Player#", i
        player = players[i]
        rad_zeroes[player['hero_id']] = 1



    for i in range(5,10):
        #print "Dire Player#", i
        player = players[i]
        dire_zeroes[player['hero_id']] = 1


    all_zeroes = []
    all_zeroes.append(rad_zeroes)
    all_zeroes.append(dire_zeroes)

    gameResults.append(radiant_team)
    gameResults.append(dire_team)
    with open('dota2games.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for zeroes in all_zeroes:
            spamwriter.writerow([zeroes])

    with open('dota2gamesResults.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(gameResults)):
            spamwriter.writerow(gameResults[i])


    #Print the hero names that are in the game ie from Zeroes to Heroes
    for zeroes in all_zeroes:
        for i,x in enumerate(zeroes):
            if x == 1:
                print (i)
                heroes = api.get_heroes()
                newheroes = heroes['heroes']
                print (newheroes[i])
    number = 0

    knn = KNN()
    #while number != 1 or number != 2 or number != 3:
     #   print ("\nChoose the Data you would like to use\n"
      #         "To view Iris Prediction,          enter 1\n"
       #        "To view Cars Prediction,          enter 2\n"
        #       "To view Breast Cancer Prediction, enter 3")

        #number = int(input("Choice: "))

        #if (number == 1):
            #irisData = datasets.load_iris()
           # trainData = irisData.data
          #  targetData = irisData.target
         #   train_system(trainData, targetData, knn)

        #not sure why but it doesnt want to load my csv
        #if (number == 2):
       #     carData = pandas.read_csv("cardata.csv")
      #      carData = carData.values
     #       trainData, targetData = carData[:, :6], carData[:, 6]
            #trainData = carData[['first', 'second', 'third', 'fourth', 'fifth', 'sixth']]
            #print (carData.values)
            #print (trainData)
            #targetData = carData['target']
    #        train_system(trainData, targetData, knn)

#      if (number == 3):
#            breastCancerData = datasets.load_breast_cancer()
 #           trainData = breastCancerData.data
  #          targetData = breastCancerData.target
   #         train_system(trainData, targetData, knn)

if __name__ == "__main__":
    main(sys.argv)

