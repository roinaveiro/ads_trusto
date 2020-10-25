import numpy as np
import pandas as pd


class ADS:
    """
    Class to simulate road
    Args:
        l (int): road lenght
    """

    def __init__(self, road):
        self.road = road
        self.N = len(road)
        self.Dir = np.ones([3,3])
        self.current_cell = 0 ## Current cell
        self.next_cell = self.road[self.current_cell + 1] ## I see next cell

        ##
    def move(self):

        ## Update knowledge
        observed_cell = self.road[self.current_cell + 2]
        self.Dir[observed_cell, self.next_cell] += 1

        self.current_cell += 1
        self.next_cell = self.road[self.current_cell + 1] 


    def predict(self):

        predictions = {}

        ## One cell ahead. This is observed
        if self.next_cell == 0:
            predictions["1"] = np.array([1,0,0])
        elif self.next_cell == 1:
            predictions["1"] = np.array([0,1,0])
        else:
            predictions["1"] = np.array([0,0,1])


        ## Two cells ahead
        if self.road[self.current_cell + 2] == 0: ## Then I see the rock
            predictions["2"] = np.array([1,0,0])
        else:
            prob = self.normalize( self.Dir[:, self.next_cell] )
            predictions["2"] = np.append( 0, self.normalize(prob[1:]) )

        ## Three cells ahead
        if self.road[self.current_cell + 3] == 0: ## Then I see the rock
            predictions["3"] = np.array([1,0,0])

        elif self.road[self.current_cell + 2] == 0: ## Previous cell was rock
            prob = self.normalize( self.Dir[:, self.current_cell + 2] )
            predictions["3"] = np.append( 0, self.normalize(prob[1:]) )
            
        else: ## previous cell either clean or puddle
            aux = self.normalize_arr( self.Dir[1:, 1:] )
            predictions["3"] = np.append(0, np.dot(aux, predictions["2"][1:].T) )

        ## Four cells ahead
        if self.road[self.current_cell + 3] == 0: ## Previous cell was rock
            predictions["4"] = self.normalize(self.Dir[:, 0])

        else: ## Previous was either puddle or clean
            predictions["4"] = np.dot( self.normalize_arr(self.Dir[:,1:]), predictions["3"][1:].T )

        ## Five cells ahead
            predictions["5"] = np.dot( self.normalize_arr(self.Dir), predictions["4"].T )

        return(predictions)


        

    def complete_road(self):
        for i in range(self.N-5):
            self.move()

    @staticmethod
    def normalize(arr):
        return arr / np.sum(arr)

    @staticmethod
    def normalize_arr(arr):
        return arr / np.sum(arr, axis=0)




