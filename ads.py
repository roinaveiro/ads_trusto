import numpy as np
import pandas as pd


class ADS:
    """
    Class to simulate road
    Args:
        l (int): road lenght
    """

    def __init__(self, road, char, driver_char, driver_state_evol):
        self.road = road
        self.N = len(road)

        # ODD and ENV variables
        self.Dir = np.ones([3,3])
        self.current_cell = 0 ## Current cell
        self.next_cell = self.road[self.current_cell + 1] ## I see CONTENT of next cell
        self.Dir[self.next_cell, self.current_cell] += 1  ## I update my knowledge

        # Driver state variables
        self.driver_char = driver_char 
        self.driver_state_evol = driver_state_evol
        self.char = char
        self.prior_driver_state = np.array([0.9, 0.1]) ## For every possible initial road state
        ## This is p(driver_state | char, road state)
        self.prob_driver_state = self.normalize(self.driver_char[str(self.char[0])].values * self.prior_driver_state)

        ##
        self.env_states = self.driver_state_evol.index.unique("Obstacle").values
        self.driver_states = self.driver_state_evol.index.unique("Current").values

        ##
    def move(self):

        ## Update environment knowledge
        observed_cell = self.road[self.current_cell + 2]
        self.Dir[observed_cell, self.next_cell] += 1

        ## Update driver state knowledge
        aux = self.driver_state_evol.xs(self.next_cell, level="Obstacle").values.T
        aux = np.dot(aux, self.prob_driver_state.T)
        self.prob_driver_state = self.normalize(self.driver_char[str(self.char[self.current_cell+1])].values * aux)

        self.current_cell += 1
        self.next_cell = self.road[self.current_cell + 1] 

    def predict_driver_state(self):

        predictions = {}
        env_pred = self.predict_env()


        ## One cell ahead.
        nstate_nobs = np.zeros( [len(self.driver_states), len(self.env_states)] )
        for i in self.env_states:
            aux = self.driver_state_evol.xs(i, level="Obstacle").values.T
            nstate_nobs[:, i] = np.dot(aux, self.prob_driver_state.T)
        predictions["1"] = np.dot(nstate_nobs, env_pred["1"])

        for k in [2,3,4,5]:
            ## Step 1
            y_bwd = self.normalize_arr( ( self.normalize_arr(self.Dir) * env_pred[str(k-1)] ).T ) 

            ## Step 2
            state_nobs = np.dot(nstate_nobs, y_bwd)

            ## Step 3
            nstate_nobs = np.zeros( [len(self.driver_states), len(self.env_states)] )
            for i in self.env_states:
                aux = self.driver_state_evol.xs(i, level="Obstacle").values.T
                nstate_nobs[:, i] = np.dot(aux, state_nobs[:,i])

            ## Step 4
            predictions[str(k)] = np.dot(nstate_nobs, env_pred[str(k)])

        return(predictions)
        

    def predict_env(self):

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


    '''
        TRASH
        
        ## Two cells ahead
        ## Step 1
        y_bwd = self.normalize_arr( ( self.normalize_arr(self.Dir) * env_pred["1"] ).T ) 

        ## Step 2
        state_nobs = np.dot(nstate_nobs, y_bwd)

        ## Step 3
        nstate_nobs = np.zeros( [len(self.driver_states), len(self.env_states)] )
        for i in self.env_states:
            aux = self.driver_state_evol.xs(i, level="Obstacle").values.T
            nstate_nobs[:, i] = np.dot(aux, state_nobs[:,i])
        
        ## Step 4
        predictions["2"] = np.dot(nstate_nobs, env_pred["2"])
        '''




