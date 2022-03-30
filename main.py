from agents1.BlindAgent import BlindAgent
from agents1.LazyAgent import LazyAgent
from agents1.LiarAgent import LiarAgent
from agents1.StrongAgent import StrongAgent
from bw4t.BW4TWorld import BW4TWorld
from bw4t.statistics import Statistics

"""
This runs a single session. You have to log in on localhost:3000 and 
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
       # {'name':'liar1', 'botclass':LiarAgent, 'settings':{}},
        #{'name': 'liar2', 'botclass': LiarAgent, 'settings': {}},
        #{'name': 'liar3', 'botclass': LiarAgent, 'settings': {}},
        #{'name':'blind', 'botclass':BlindAgent, 'settings':{}},
        # {'name':'lazy3', 'botclass': LazyAgent, 'settings':{}},
        # {'name': 'lazy1', 'botclass': LazyAgent, 'settings': {}},
        # {'name': 'lazy2', 'botclass': LazyAgent, 'settings': {}},
        # {'name': 'lazy4', 'botclass': LazyAgent, 'settings': {}},
        # {'name': 'lazy5', 'botclass': LazyAgent, 'settings': {}},
        {'name': 'strong1', 'botclass': StrongAgent, 'settings': {}},
        {'name': 'strong2', 'botclass': StrongAgent, 'settings': {}},
        {'name': 'strong3', 'botclass': StrongAgent, 'settings': {}},

        #{'name': 'humna', 'botclass' : Human, 'settings':{}},
        ]

    print("Started world...")
    world=BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))
