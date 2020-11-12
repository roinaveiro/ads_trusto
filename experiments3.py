import numpy as np
import pandas as pd
from simulator import simulator
from ads import ADS

N_sim = 1000
results = []


road_dynamics = pd.read_csv("data/road_state_evol", index_col=0, delim_whitespace=True)
driver_dynamics = pd.read_csv("data/driver_state_evol", delim_whitespace=True)
driver_dynamics.set_index(["Current", "Obstacle"], inplace=True)
driver_char = pd.read_csv("data/driver_char", index_col=0, delim_whitespace=True)

grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
results = []

for i in range(N_sim):
    print(i)
    for j, pr in enumerate(grid):

        road_dynamics.loc[2][1] = pr
        road_dynamics.loc[2][2] = 1.0 - (road_dynamics.loc[2][0] + road_dynamics.loc[2][1]) 

        sim = simulator(1000, [road_dynamics, driver_dynamics, driver_char])
        env = sim.simulate_environment()
        road = env["road"]
        char = env["driver_char"]
        driver = env["driver"]

        ads = ADS(road, char, driver, driver_char, driver_dynamics)
        ads.complete_road()
        dirr = ads.get_info()
        dirr['n_exp'] = i
        dirr['pr_puddle'] = pr
        results.append(dirr)
 

df = pd.DataFrame(results)
df.to_csv("results/sim3_prop_puddle_1000.csv", index=False)

