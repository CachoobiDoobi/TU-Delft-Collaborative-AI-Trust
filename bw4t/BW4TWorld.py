import numpy as np
import random
import pathlib
import os
from typing import Final, List
from matrx.actions.move_actions import MoveEast, MoveSouth, MoveWest 
from matrx.actions import MoveNorth, OpenDoorAction, CloseDoorAction 
from matrx.grid_world import GridWorld, DropObject, GrabObject, AgentBody 
from matrx import WorldBuilder 
from matrx.world_builder import RandomProperty
from matrx.agents import SenseCapability 
from matrx.utils import get_room_locations
from bw4t.BW4TBlocks import CollectableBlock, GhostBlock
from agents1.BW4THuman import Human
from bw4t.CollectionGoal import CollectionGoal
from bw4t.BW4TLogger import BW4TLogger
from bw4t.BW4THumanBrain import HumanBrain

DEFAULT_WORLDSETTINGS: dict={
    'deadline': 3000, # Ticks after which world terminates anyway 
    'tick_duration': 0, # Set to 0 for fastest possible runs.
    'random_seed': 1,
    'verbose': False,
    'matrx_paused':True,
    'run_matrx_api':True, # If you want to allow web connection
    'run_matrx_visualizer':True, # if you want to allow web visualizer

    'key_action_map': {  # For the human agents
        'w': MoveNorth.__name__,
        'd': MoveEast.__name__,
        's': MoveSouth.__name__,
        'a': MoveWest.__name__,
        'q': GrabObject.__name__,
        'e': DropObject.__name__,
        'r': OpenDoorAction.__name__,
        'f': CloseDoorAction.__name__,
    },
    'room_size' : (6, 4),  # width, height
    'nr_rooms' : 9, # total number of rooms.
    'rooms_per_row':3, #number of rooms per row.
    'average_blocks_per_room': 2,
    'block_shapes': [0, 1, 2], # possible shapes of the blocks
    'block_colors': ['#0008ff', '#ff1500', '#0dff00'], #possible colors of blocks
    'room_colors': ['#0008ff', '#ff1500', '#0dff00'],
    'wall_color': "#8a8a8a",
    'drop_off_color': "#878787",
    'block_size' : 0.5,
    'nr_drop_zones':  1, # All code assumes this is 1, don't change this.
    'nr_blocks_needed':  3, # nr of drop tiles/target blocks
    'hallway_space': 2, # width, height of corridors

    'agent_sense_range':  2,  # the range with which agents detect other agents
    'block_sense_range': 1,  # the range with which agents detect blocks
    'other_sense_range':  np.inf , # the range with which agents detect other objects (walls, doors, etc.)
    'agent_memory_decay': 5,  # we want to memorize states for seconds / tick_duration ticks
    'fov_occlusion' : True # true if walls block vision. Not sure if this works at all.
    
}


class BW4TWorld:
    '''
    Creates a single GridWorld to be run. 
    The idea was that this extends GridWorld, however
    it seems we need to create these through the WorldBuilder
    and therefore this approach seems not possible.
    Instead, this only supports the 'run' function and
    internally creates the gridworld using WorldBuilder.
    
    '''
    def __init__(self, agents:List[dict], worldsettings:dict=DEFAULT_WORLDSETTINGS):
        '''
           @param agents a list like 
            [
            {'name':'agent1', 'botclass':PatrollingAgent, 'settings':{'slowdown':3}},
            {'name':'human1', 'botclass':Human, 'settings':{'slowdown':1}}
            ]
            Names must all be unique.
            Check BW4TBrain for more on the agents specification.
        '''
        self._worldsettings=worldsettings;
        self._agents=agents
        
        np.random.seed(worldsettings['random_seed'])
        world_size = self.world_size()
    
        # Create the goal
        goal = CollectionGoal(worldsettings['deadline'])
    
        # Create our world builder
        self._builder = WorldBuilder(shape=world_size, tick_duration=worldsettings['tick_duration'], 
           random_seed=worldsettings['random_seed'], 
           run_matrx_api=worldsettings['run_matrx_api'],
           run_matrx_visualizer=worldsettings['run_matrx_visualizer'], 
           verbose=worldsettings['verbose'], simulation_goal=goal)
        
        self._builder.api_info['_matrx_paused']=worldsettings['matrx_paused']
    
        # Add the world bounds (not needed, as agents cannot 'walk off' the grid, but for visual effects)
        self._builder.add_room(top_left_location=(0, 0), width=world_size[0], height=world_size[1], name="world_bounds")
        room_locations = self._addRooms()
        self._addBlocks(room_locations)
        self._addDropOffZones(world_size)
    
        # Add the agents and human agents to the top row of the world
        self._addAgents()
        
        #media_folder = os.path.dirname(os.path.join(os.path.realpath(__file__), "media"))
        media_folder = pathlib.Path().resolve()
        self._builder.startup(media_folder=media_folder)
        self._builder.add_logger(BW4TLogger, save_path='.')

        self._gridworld = self._builder.worlds(nr_of_worlds=1).__next__()

    def run(self):
        '''
        run the world till termination
        '''
        self._gridworld.run(self._builder.api_info)
        return self
        
    def getLogger(self)->BW4TLogger:
        '''
        @return the logger. We assume there is only 1: BW4TLogger
        '''
        return self._gridworld._GridWorld__loggers[0]
        
        
    def world_size(self):
        '''
        returns (width,height) (number of tiles)
        '''
        worldsettings=self._worldsettings
        nr_room_rows = np.ceil(worldsettings['nr_rooms'] / worldsettings['rooms_per_row'])
    
        # calculate the total width
        world_width = max(worldsettings['rooms_per_row'] * worldsettings['room_size'][0] + 2 * worldsettings['hallway_space'],
                          (worldsettings['nr_drop_zones'] + 1) * worldsettings['hallway_space'] + worldsettings['nr_drop_zones']) + 2
    
        # calculate the total height
        world_height = nr_room_rows * worldsettings['room_size'][1] + (nr_room_rows + 1) * worldsettings['hallway_space'] + worldsettings['nr_blocks_needed'] + 2
    
        return int(world_width), int(world_height)   
    
        
    def _addBlocks(self, room_locations):
        '''
        Add blocks to all given room locations
        '''
        for room_name, locations in room_locations.items():
            for loc in locations:
                # Get the block's name
                name = f"Block in {room_name}"
    
                # Get the probability for adding a block so we get the on average the requested number of blocks per room
                prob = min(1.0, self._worldsettings['average_blocks_per_room'] / len(locations))
    
                # Create a MATRX random property of shape and color so each block varies per created world.
                # These random property objects are used to obtain a certain value each time a new world is
                # created from this builder.
                colour_property = RandomProperty(values=self._worldsettings['block_colors'])
                shape_property = RandomProperty(values=self._worldsettings['block_shapes'])
    
                # Add the block; a regular SquareBlock as denoted by the given 'callable_class' which the
                # builder will use to create the object. In addition to setting MATRX properties, we also
                # provide a `is_block` boolean as custom property so we can identify this as a collectible
                # block.
                self._builder.add_object_prospect(loc, name, 
                    callable_class=CollectableBlock, probability=prob,
                    visualize_shape=shape_property, visualize_colour=colour_property,
                    block_size=self._worldsettings['block_size'])

    def _addAgents(self):
        '''
        Add bots as specified, starting top left corner. 
        All bots have the same sense_capability.
        '''
        sense_capability = SenseCapability({
            AgentBody: self._worldsettings['agent_sense_range'],
            CollectableBlock: self._worldsettings['block_sense_range'],
            None: self._worldsettings['other_sense_range']})
    
        loc = (0,1) # agents start in horizontal row at top left corner.
        team_name = "Team 1" # currently this supports 1 team 
        for agent in self._agents:
            brain = agent['botclass'](agent['settings'])
            loc = (loc[0] + 1, loc[1])
            if agent['botclass']==Human:
                self._builder.add_human_agent(loc, brain,
                team=team_name, name=agent['name'],
                key_action_map=self._worldsettings['key_action_map'],
                sense_capability=sense_capability, visualize_shape=2, visualize_colour='#FFFF00')
            else:
                self._builder.add_agent(loc, brain, 
                team=team_name, name=agent['name'], 
                sense_capability=sense_capability, visualize_shape=1, visualize_colour=self._worldsettings['block_colors'][random.randint(0,2)])
     
    def _addRooms(self):
        '''
        @return room locations
        '''
        room_locations = {}
        for room_nr in range(self._worldsettings['nr_rooms']):
            room_top_left, door_loc = self.get_room_loc(room_nr)
    
            # We assign a simple random color to each room. Not for any particular reason except to brighting up the place.
            room_color = random.choice(self._worldsettings['room_colors'])
                                       
    
            # Add the room
            room_name = f"room_{room_nr}"
            self._builder.add_room(top_left_location=room_top_left, 
                 width=self._worldsettings['room_size'][0], 
                 height=self._worldsettings['room_size'][1], name=room_name,
                 door_locations=[door_loc], 
                 wall_visualize_colour=self._worldsettings['wall_color'],
                 with_area_tiles=True, area_visualize_colour=room_color, 
                 area_visualize_opacity=0.1)
    
            # Find all inner room locations where we allow objects (making sure that the location behind to door is free)
            room_locations[room_name] = get_room_locations(room_top_left, self._worldsettings['room_size'][0], self._worldsettings['room_size'][1])
    
        return room_locations       
    
        
    def get_room_loc(self,room_nr):
        '''
        @return room location (room_x, room_y), (door_x, door_y) for given room nr
        '''
        row = np.floor(room_nr / self._worldsettings['rooms_per_row'])
        column = room_nr % self._worldsettings['rooms_per_row']
    
        # x is: +1 for the edge, +edge hallway, +room width * column nr, +1 off by one
        room_x = int(1 + self._worldsettings['hallway_space'] + (self._worldsettings['room_size'][0] * column) )
    
        # y is: +1 for the edge, +hallway space * (nr row + 1 for the top hallway), +row * room height, +1 off by one
        room_y = int(1 + self._worldsettings['hallway_space'] * (row + 1) + row * self._worldsettings['room_size'][1] + 1)
    
        # door location is always center bottom
        door_x = room_x + int(np.ceil(self._worldsettings['room_size'][0] / 2))
        door_y = room_y + self._worldsettings['room_size'][1] - 1
    
        return (room_x, room_y), (door_x, door_y)
    
    
    def _addDropOffZones(self, world_size):
        x = int(np.ceil(world_size[0] / 2)) - \
            (int(np.floor(self._worldsettings['nr_drop_zones'] / 2)) * \
                (self._worldsettings['hallway_space'] + 1))
        y = world_size[1] - 1 - 1  # once for off by one, another for world bound
        for nr_zone in range(self._worldsettings['nr_drop_zones']):
            # Add the zone's tiles. Area tiles are special types of objects in MATRX that simply function as
            # a kind of floor. They are always traversable and cannot be picked up.
            self._builder.add_area((x, y - self._worldsettings['nr_blocks_needed'] + 1), 
                 width=1, height=self._worldsettings['nr_blocks_needed'], 
                 name=f"Drop off {nr_zone}",
                 visualize_colour=self._worldsettings['drop_off_color'], 
                 drop_zone_nr=nr_zone, is_drop_zone=True, 
                 is_goal_block=False, is_collectable=False)
    
            # Go through all needed blocks
            for nr_block in range(self._worldsettings['nr_blocks_needed']):
                # Create a MATRX random property of shape and color so each world contains different blocks to collect
                colour_property = RandomProperty(values=self._worldsettings['block_colors'])
                shape_property = RandomProperty(values=self._worldsettings['block_shapes'])
    
                # Add a 'ghost image' of the block that should be collected. This can be seen by both humans and agents to
                # know what should be collected in what order.
                loc = (x, y - nr_block)
                self._builder.add_object(loc, 
                   name="Collect Block", callable_class=GhostBlock,
                   visualize_colour=colour_property, visualize_shape=shape_property,
                   drop_zone_nr=nr_zone, block_size=self._worldsettings['block_size'])
                
            # Change the x to the next zone
            x = x + self._worldsettings['hallway_space'] + 1
