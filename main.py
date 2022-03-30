from agents1.StrongAgent import StrongAgent
from agents1.LiarAgent import LiarAgent
from agents1.BlindAgent import BlindAgent
from bw4t.BW4TWorld import BW4TWorld
from bw4t.statistics import Statistics
from agents1.BW4TBaselineAgent import BaseLineAgent
from agents1.BW4THuman import Human
from agents1.LazyAgent import LazyAgent
# from agents1.StrongAgentRefactored import StrongAgentRefactored

import json
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
        # {'name':'liar', 'botclass': LiarAgent, 'settings':{}},
        {'name':'lazy', 'botclass': LazyAgent, 'settings':{}},
        # {'name': 'strong', 'botclass': StrongAgentRefactored, 'settings': {}},
        {'name': 'blind', 'botclass': BlindAgent, 'settings': {}},
    ]

    print("Started world...")
    world=BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))

    ## TRUST TEST + PLOT ###

    trusts = {}
    for agent in agents:
        trusts[agent['name']] = {}

    # Read from each csv
    for agent in agents:
        file_name = agent['name'] + '_trust.csv'
        with open(file_name, newline='') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                if row:
                    trusts[agent['name']][row[0]] = float(row[4])

    read = open("trust_test.txt", "r")
    empty = len(read.readlines()) == 0

    # Write to trust_test data from all csv's
    f = open("trust_test.txt", "a")
    if not empty:
        f.write("\n")
    f.write(json.dumps(trusts))
    f.close()

    # Read new file
    dictbig = {}
    for agent in agents:
        dictbig[agent['name']] = {}
        for agent2 in agents:
            if agent2['name'] != agent['name']:
                dictbig[agent['name']][agent2['name']] = []

    f = open("trust_test.txt", "r")
    lines = f.readlines()
    for line in lines:
        dict = json.loads(line)
        for key in dict:
            for key2 in dict[key]:
                dictbig[key][key2].append(dict[key][key2])

    # Prepare array with results (to plot)
    toplot = {}
    for agent in agents:
        toplot[agent['name']] = []

    for key in dictbig:
        for key2 in dictbig[key]:
            if toplot[key2] == []:
                toplot[key2] = dictbig[key][key2]
            else:
                toplot[key2] = np.add(toplot[key2], dictbig[key][key2])

    number_observations = len(next(iter((toplot.items())))[1]) + 1
    rounds = []
    for i in range(1, number_observations):
        rounds.append(i)
    colors = ['red', 'blue', 'green', 'yellow'] # add more colors if more then 4 agents

    i = 0
    for key in toplot:
        toplot[key] = np.array(toplot[key]) / (len(agents) - 1)
        plt.plot(rounds, toplot[key], color = colors[i], label = key)
        i = i + 1

    plt.legend()
    plt.show()

    file = open("average_trust_result.txt", "w")
    file.write(str(toplot))
    file.close()

