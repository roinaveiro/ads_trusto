import numpy as np
import pandas as pd


class simulator:
    """
    Class to simulate road
    Args:
        l (int): road lenght
    """

    def __init__(self, l, dynamics=None):
        self._l = l
        ##
        if dynamics is None:
            self._road_dynamics = pd.read_csv("data/road_state_evol", index_col=0, delim_whitespace=True)
            ##
            self._driver_dynamics = pd.read_csv("data/driver_state_evol", delim_whitespace=True)
            self._driver_dynamics.set_index(["Current", "Obstacle"], inplace=True)
            ##
            self._driver_char = pd.read_csv("data/driver_char", index_col=0, delim_whitespace=True)
        else:
            self._road_dynamics = dynamics[0]
            ##
            self._driver_dynamics = dynamics[1]
            ##
            self._driver_char = dynamics[2]

    def simulate_road(self):

        road = np.empty(self._l, dtype=int)
        road[0] = 2

        for i in range(1, self._l):
            p = self._road_dynamics.loc[ road[i-1] ]
            road[i] = np.random.choice(self._road_dynamics.columns, p = p)

        return road

    def simulate_driver_state(self, road):

        driver = np.empty(self._l, dtype=int)
        driver[0] = 0

        for i in range(1, self._l):
            p = self._driver_dynamics.loc[ (driver[i-1], road[i]) ]
            driver[i] = np.random.choice(self._driver_dynamics.columns, p = p)

        return driver
    
    def simulate_driver_char(self, driver):

        driver_char = np.empty(self._l, dtype=int)
        
        for i in range(self._l):
            p = self._driver_char.loc[ driver[i] ]
            driver_char[i] = np.random.choice(self._driver_char.columns.astype(int), p = p)
        
        return driver_char

    def simulate_environment(self):
        
        road          = self.simulate_road()
        driver        = self.simulate_driver_state(road)
        driver_char   = self.simulate_driver_char(driver)

        return {"road" : road, "driver" : driver, "driver_char" : driver_char}