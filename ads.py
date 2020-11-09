import numpy as np
import pandas as pd


class ADS:
    """
    Class to simulate road
    Args:
        l (int): road lenght
    """

    def __init__(self, road, char, driver, driver_char, driver_state_evol):

        # Road details and driver state
        self.road = road
        self.N = len(road)
        self.driver = driver

        # Init ads
        self.current_cell = 0 ## Current cell
        self.next_cell = self.road[self.current_cell + 1] ## I see CONTENT of next cell
        # self.mode = "AUTON" ## Start with auton

        # ODD and ENV variables
        self.Dir = np.ones([3,3])
        self.Dir[self.next_cell, self.current_cell] += 1  ## I update my knowledge, coz I see the next cell

        # Driver state variables
        self.driver_char = driver_char 
        self.driver_state_evol = driver_state_evol
        self.char = char
        self.prob_driver_state_arr = np.zeros(self.N)
        self.prior_driver_state = np.array([0.9, 0.1]) ## For every possible initial road state
        ## This is p(driver_state | char, road state)
        self.prob_driver_state = self.normalize(self.driver_char[str(self.char[0])].values \
            * self.prior_driver_state)
        self.prob_driver_state_arr[0] = self.prob_driver_state[1]

        ## Relevant parameters
        self.env_states = self.driver_state_evol.index.unique("Obstacle").values
        self.driver_states = self.driver_state_evol.index.unique("Current").values

        # Trajectory planning and utilities
        self.v_auton  = {0:0, 1:2, 2:3} # Obstacle: velocity (AUTON mode)
        self.v_manual = {0:0, 1:1, 2:4} # Obstacle: velocity (MANNUAL mode)
        self.u_d = {0:0.0, 1:0.1, 2:0.2, 3:0.3, 4:0.5}

        # Warnings
        self.driver_state_threshold = 0.85
        self.env_state_threshold_rock = 0.15
        self.env_state_threshold_puddle = 0.25
        ##
        self.rock_warnings = np.zeros(self.N) + 100
        self.puddle_warnings = np.zeros(self.N) + 100
        self.state_warnings = np.zeros(self.N) + 100

        # Issue first warnings
        ## Forecasts - Driver state
        state_pred = self.predict_driver_state()
        self.state_pred = np.vstack(list(state_pred.values()))

        ## Forecasts - Environment state
        env_pred = self.predict_env()
        self.env_pred = np.vstack(list(env_pred.values()))

        s,r,p = self.issue_warnings()
        self.rock_warnings[self.current_cell] = r
        self.puddle_warnings[self.current_cell] = p
        self.state_warnings[self.current_cell] = s

        # Decisions made and modes
        self.modes = np.array(['AUTON' for _ in range(self.N)], dtype=object)
        self.decision_auton = np.zeros(self.N) + 100
        self.decision_manual = np.zeros(self.N) + 100
        self.decision_manual_aware = np.zeros(self.N) + 100
        self.decision_manual_dist = np.zeros(self.N) + 100
        ##
        self.decision_auton[self.current_cell] = self.trajectory_planning("AUTON")[0]
        self.decision_manual_aware[self.current_cell] = self.trajectory_planning("MANUAL_AWARE")[0]
        self.decision_manual_dist[self.current_cell] = self.trajectory_planning("MANUAL_DIST")[0]
        self.decision_manual[self.current_cell] = self.decision_manual_aware[self.current_cell]

        # Utilities attained
        self.utilities = np.zeros(self.N)
        self.utilities[self.current_cell] = self.compute_cell_utility('AUTON', 
            self.decision_auton[self.current_cell])

        # Counters
        self.RtI = 0
        self.prop_RtI = 0
        self.emergency = 0
        self.crashes = 0
        self.skids = 0


        ##
    def move(self):

        ## Update environment knowledge
        observed_cell = self.road[self.current_cell + 2]
        self.Dir[observed_cell, self.next_cell] += 1

        ## Update driver state knowledge
        aux = self.driver_state_evol.xs(self.next_cell, level="Obstacle").values.T
        aux = np.dot(aux, self.prob_driver_state.T)
        self.prob_driver_state = self.normalize(self.driver_char\
            [str(self.char[self.current_cell+1])].values * aux)

        # Move 
        self.current_cell += 1
        self.next_cell = self.road[self.current_cell + 1] 
        self.prob_driver_state_arr[self.current_cell] = self.prob_driver_state[1]

        ## Forecasts - Driver state
        state_pred = self.predict_driver_state()
        self.state_pred = np.vstack(list(state_pred.values()))

        ## Forecasts - Environment state
        env_pred = self.predict_env()
        self.env_pred = np.vstack(list(env_pred.values()))

        # Issue warnings
        s,r,p = self.issue_warnings()
        self.rock_warnings[self.current_cell] = r
        self.puddle_warnings[self.current_cell] = p
        self.state_warnings[self.current_cell] = s

        # Make speed decisions and trajectory planning
        traj_auton = self.trajectory_planning("AUTON")
        traj_manual_aware = self.trajectory_planning("MANUAL_AWARE")
        traj_manual_dist = self.trajectory_planning("MANUAL_DIST")

        self.traj_plan_auton = traj_auton[1:]
        self.traj_plan_manual_aware = traj_manual_aware[1:]
        self.traj_plan_manual_dist = traj_manual_dist[1:]

        self.decision_auton[self.current_cell] = traj_auton[0]
        self.decision_manual_aware[self.current_cell] = traj_manual_aware[0]
        self.decision_manual_dist[self.current_cell] = traj_manual_dist[0]

        if self.driver[self.current_cell - 1] == 0:
            self.decision_manual[self.current_cell] = \
                self.decision_manual_aware[self.current_cell] * (1-self.driver[self.current_cell]) \
                    + self.decision_manual_aware[self.current_cell-1] * self.driver[self.current_cell]
        else:
            self.decision_manual[self.current_cell] = \
                self.decision_manual_dist[self.current_cell -1]

        if self.modes[self.current_cell] == "AUTON":
            self.eval_RtI()
            self.utilities[self.current_cell] = self.compute_cell_utility('AUTON', 
                self.decision_auton[self.current_cell])

        else:

            self.utilities[self.current_cell] = self.compute_cell_utility('MANUAL', 
                self.decision_manual[self.current_cell])

            self.counter_manual += 1 
            if self.prob_driver_state[1] > 0.75 or self.counter_manual >= 10:
                self.modes[self.current_cell + 1] = "AUTON"
            else:
                self.modes[self.current_cell + 1] = "MANUAL"
            


    def trajectory_planning(self, mode):

        if mode == "AUTON":
            max_env_pred = np.argmax(self.env_pred, axis=1)
            trajectory = np.vectorize(self.v_auton.get)(max_env_pred)
            return np.append(self.v_auton[self.road[self.current_cell]], trajectory)

        elif mode == "MANUAL_AWARE":
            env_pred = self.road[self.current_cell:self.current_cell+6]
            return np.vectorize(self.v_manual.get)(env_pred)

        else:
            env_pred = np.append(self.road[self.current_cell], np.array([2,2,2,2,2]))

            if self.road[self.current_cell + 1] == 0: 
                env_pred[1] = 0

            if self.road[self.current_cell + 2] == 0:
                env_pred[2] = 0
                
            return np.vectorize(self.v_manual.get)(env_pred)

            
            

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
            prob = self.normalize( self.Dir[:, self.road[self.current_cell + 2] ] )
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


    def issue_warnings(self):

        ## Driver state warnings
        w_state = np.any(self.state_pred[:,1] > self.driver_state_threshold)

        ## Env state warning Rock
        w_rock = np.any(self.env_pred[3:,0] > self.env_state_threshold_rock)

        ## Env state warning Puddle
        w_puddle = np.any(self.env_pred[1:,1] > self.env_state_threshold_puddle)

        return int(w_state), int(w_rock), int(w_puddle)

    def compute_cell_utility(self, mode, d):

        if mode == "AUTON":

            if self.road[self.current_cell] == 0: #If rock
                if d!=0:
                    self.crashes += 1
                    ut_obstacle = -100 
                else:
                    ut_obstacle = 0

            elif self.road[self.current_cell] == 1: #If puddle

                if d != 3:
                    ut_obstacle = 0
                else:
                    if np.random.random() < 0.95: # Skid!!
                        self.skids += 1
                        ut_obstacle = -10
                    else:
                        ut_obstacle = 0           

            else:
                ut_obstacle = 0

            ut = 0.1 + self.u_d[d] + ut_obstacle
            
        else:

            if self.road[self.current_cell] == 0: #If rock
                if d!=0:
                    self.crashes += 1
                    ut_obstacle = -100 
                else:
                    ut_obstacle = 0

            elif self.road[self.current_cell] == 1: #If puddle

                if d == 2:

                    if np.random.random() < 0.5: # Skid!!
                        self.skids += 1
                        ut_obstacle = -10
                    else:
                        ut_obstacle = 0

                elif d==3:

                    if np.random.random() < 0.80: # Skid!!
                        self.skids += 1
                        ut_obstacle = -10
                    else:
                        ut_obstacle = 0

                elif d==4:

                    if np.random.random() < 0.85: # Skid!!
                        self.skids += 1
                        ut_obstacle = -10
                    else:
                        ut_obstacle = 0

                else:
                    ut_obstacle = 0
                               
                        
            else:
                ut_obstacle = 0.0

            ut = self.u_d[d] + ut_obstacle

        return ut


    def compute_cell_exp_utility(self, mode, env_pred, d):

        if mode == "AUTON":
            eut = 0.1 + self.u_d[d] + env_pred[1] * (-10*0.95 if d==3 else 0) + \
                env_pred[0] * (-100 if d!=0 else 0)
        else:
            eut = self.u_d[d] + env_pred[1] * -10 * \
                (0.5 if d==2 else 0.8 if d==3 else 0.85 if d==4 else 0) + \
                env_pred[0] * (-100 if d!=0 else 0)

        return eut

    def evaluate_driving_modes(self):

        eut_auton = 0
        eut_manual_aware = 0
        eut_manual_dist = 0

        for i in range(5):

            eut_auton += self.compute_cell_exp_utility("AUTON", self.env_pred[i,:], self.traj_plan_auton[i] )
            eut_manual_aware += self.compute_cell_exp_utility("MANUAL_AWARE", self.env_pred[i,:], self.traj_plan_manual_aware[i] )
            eut_manual_dist += self.compute_cell_exp_utility("MANUAL_DIS", self.env_pred[i,:], self.traj_plan_manual_dist[i] )


        return eut_auton, self.prob_driver_state[0] * eut_manual_aware + self.prob_driver_state[1] * eut_manual_dist

    def eval_RtI(self):
        if self.rock_warnings[self.current_cell] == 1 or self.puddle_warnings[self.current_cell] == 1:
            self.prop_RtI += 1
            eut_auton, eut_manual = self.evaluate_driving_modes()
            if eut_manual > eut_auton:
                self.RtI += 1
                if self.prob_driver_state[1] > 0.95:
                    self.emergency +=1
                else:
                    self.counter_manual = 0
                    if self.driver[self.current_cell] == 1: 
                        self.modes[self.current_cell + 1] = "MANUAL"
                    else:
                        self.modes[self.current_cell + 3] = "MANUAL"

                    if self.prob_driver_state[1] > 0.5:
                        self.state_warnings[self.current_cell] += 1



    def complete_road(self):
        for i in range(self.N-6):
            self.move()

    def get_info(self):

        info = {}
        info["prop_manual"] = np.sum(self.modes[:self.N-5] == "MANUAL") / (self.N-5)
        info["n_RtI"] = self.RtI
        info["n_emergency"] = self.emergency
        info["prop_rejected_RtI"] = (self.prop_RtI - self.RtI) / self.prop_RtI
        condition = self.modes == "MANUAL"
        info["avg_len_man"] = np.mean( np.diff(np.where(np.concatenate(([condition[0]],
                                     condition[:-1] != condition[1:],
                                     [True])))[0])[::2] )
                                    
        info["state_warnings"] = np.sum(self.state_warnings[:self.N-5] != 0)
        info["rock_warnings"] = np.sum(self.rock_warnings == 1)
        info["puddle_warnings"] = np.sum(self.puddle_warnings == 1)
        info["crashes"] = self.crashes
        info["skids"] = self.skids
        info["utility"] = np.mean(self.utilities[:self.N-5])

        return info
        

    @staticmethod
    def normalize(arr):
        return arr / np.sum(arr)

    @staticmethod
    def normalize_arr(arr):
        return arr / np.sum(arr, axis=0)


    '''
        TODO_
     
     
    '''




