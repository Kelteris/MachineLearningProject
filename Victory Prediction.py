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
from sklearn.cross_validation import train_test_split as tts
import sklearn.metrics.scorer


import csv

from sklearn.neighbors import KNeighborsClassifier


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
    print ("The system correctly predicted ", value_correct, " of ", test_targets.size,
         ". \nThe system was able to correctly predict ",
         "{0:.2f}% of the time!".format(100 * (value_correct / test_targets.size)))



def train_system(data, target, classifier):
    #random.shuffle(iris.data)
    testAmount = float(0.3)
    timesShuffled = 15
    k = getKAmount()
    '''
    training_set = []
    testing_set = []
    count = 0
    seventy_percent = len(data) * .7
    for i in data:
        if count <= seventy_percent:
            training_set.append(i)
            count += 1
        else:
            testing_set.append(i)

    print len(training_set)
    print len(testing_set)
    '''
    train_data, test_data, train_target, test_target = tts(data, target, test_size = testAmount, random_state = timesShuffled)

    classifier.train(train_data, train_target)
    get_accuracy(classifier.predict(train_data, train_target, test_data, k), test_target)

def createcsv(HeroList):
    api = dota2api.Initialise("7FCC616D07990B76386BCE8AB2F51B32")
    gameResults = []
    all_zeroes = []
    sequence_num = 2300000000
    while(len(gameResults) < 25000):
        matches = api.get_match_history_by_seq_num(start_at_match_seq_num=sequence_num, matches_requested=100)


        for mat in matches['matches']:
            match_id = mat['match_id']
            if(any(mat['lobby_type'] == x for x in [8,3, -1, 4, 1])):
                continue

            print ('Lobby: ',  mat['lobby_type'])
            match = api.get_match_details(match_id=match_id)
            print (match_id, " Match win:")
            # 'radiant_win' says if the radiant team won. if false dire team won. (whatever that means)
            print (match['radiant_win']), (match['duration'])
            players = (match['players'])

            rad_zeroes = [0] * 114
            dire_zeroes = [0] * 114

            radiant_team = match['radiant_win']
            dire_team = not(radiant_team)

            if(radiant_team):
                radiant_team = 1
                dire_team = 0
            else:
                radiant_team = 0
                dire_team = 1

            # these are the match players
            for i in range(0, 5):
            # print "Radiant Player#", i
                player = players[i]
                if player['hero_id'] == 0:
                    continue
                print (player['hero_id'])
                print player['hero_name']
                rad_zeroes[player['hero_id']] = 1

            for i in range(5, 10):
                # print "Dire Player#", i
                if player['hero_id'] == 0:
                    continue
                player = players[i]
                dire_zeroes[player['hero_id']] = 1

            all_zeroes.append(rad_zeroes)
            all_zeroes.append(dire_zeroes)

            gameResults.append(radiant_team)
            gameResults.append(dire_team)
            # END OF MATCH FOR LOOP
        print ("Does this work?", match['match_seq_num'])
        sequence_num = (match['match_seq_num'] + 1)
        #WHILE LOOP END

    for res in gameResults:
        print res
    with open('dota2games.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for zeroes in all_zeroes:
            spamwriter.writerow([zeroes])


    '''
    with open('dota2gamesResults.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(gameResults)
    '''
    with open('dota2gamesResults2.csv', 'wb') as csvfile:
        spamwriter2 = csv.writer(csvfile, delimiter=' ',
                                quotechar = '|', quoting = csv.QUOTE_MINIMAL)
        for res in gameResults :
            spamwriter2.writerow([res])

    # Print the hero names that are in the game ie from Zeroes to Heroes

    for i in range(0, len(rad_zeroes)):
        if rad_zeroes[i] == 1:
            print (i)
            #eroes = api.get_heroes()
            #HeroList = heroes['heroes']
            thisguy = HeroList[i-1]
            print (thisguy)#['localized_name'])




def main(argv):
    HeroList=[]
    PlayerList=[]
    ResultsList=[]
    with open('result.csv', 'rb') as csvfile:
       spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
       for row in spamreader:
            HeroList.append(row)

    #with open('dota2games.csv', 'rb') as csvfile:
    #    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #    for row in spamreader:
    #        PlayerList.append(row)


    createcsv(HeroList)

    '''
    #knn = KNN()
    train_hero_data = pandas.read_csv("dota2games.csv")
    print(train_hero_data)
    target_hero_data = pandas.read_csv("dota2gamesResults.csv")
    print(target_hero_data)
    data = train_hero_data_values = train_hero_data.values
    print ("data:",data)
    target = target_hero_data_values = target_hero_data.values
    print ("target:",target)
    timesShuffled = 3
    testAmount = .3
    #zippy = zip(trainData, targetData)
    #train_system(trainData, targetData, knn)

    train_data, test_data, train_target, test_target = tts(data, target, test_size=testAmount,
                                                           random_state=timesShuffled)
    print(train_data)
    print(train_target)
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(train_data,train_target)
    predictions = classifier.predict(test_data)
    print(predictions)
    '''
    #value_correct = 0
    #for i in range(test_target.size):
     #   value_correct += predictions[i] == test_target[i]

    #print ("The system correctly predicted ", value_correct, " of ", test_target.size,
     #      ". \nThe system was able to correctly predict ",
      #     "{0:.2f}% of the time!".format(100 * (value_correct / test_target.size)))


    #number = 0
    #knn = KNN()




if __name__ == "__main__":
    main(sys.argv)

