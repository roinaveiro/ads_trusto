import numpy as np
import pandas as pd
from simulator import simulator
from ads import ADS


sim = simulator(1000)
road = sim.simulate_environment()["road"]

ads = ADS(road)

print("hola")