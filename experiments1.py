import numpy as np
import pandas as pd
from simulator import simulator
from ads import ADS

N_sim = 10
results = []

for i in range(N_sim):
    print(i)
    sim = simulator(1000)
    env = sim.simulate_environment()
    road = env["road"]
    char = env["driver_char"]
    driver = env["driver"]

    driver_state_evol = pd.read_csv("data/driver_state_evol", delim_whitespace=True)
    driver_state_evol.set_index(["Current", "Obstacle"], inplace=True)
    driver_char = pd.read_csv("data/driver_char", index_col=0, delim_whitespace=True)

    ads = ADS(road, char, driver, driver_char, driver_state_evol)
    ads.complete_road()

    results.append(ads.get_info())

df = pd.DataFrame(results)
df.to_csv("results/sim4_less.csv", index=False)