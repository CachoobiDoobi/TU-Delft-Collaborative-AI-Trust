import enum
import random
import numpy as np
from typing import Dict

from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import DropObject
from matrx.actions.object_actions import GrabObject
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from agents1.Util import Util

from matrx.messages.message import Message
import re
import json
import csv
import os

from bw4t.BW4TBrain import BW4TBrain


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3
    SEARCH_ROOM = 4
    MOVING_BLOCK = 5
    START = 6
    GRAB = 7
    MOVE_TO_OBJECT = 8
    STOP = 9
    SEARCH_RANDOM_ROOM = 10
    PLAN_PATH_TO_OBJECT = 11
    DROP_OBJECT = 12


# Problems
# sometimes agents get stuck on delviered goal blocks, spamming pick up goal block
# agents only open doors without searching them if all doors are open

class LazyAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.START
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._door = None
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._objects = []
        self._goal_objects_found = []
        self._goal_objects = []
        self._goal_object_delivered = []
        self._current_obj = None
        self._can_be_lazy = True
        self._trust = {}
        self._arrayWorld = None
        self.read_trust()

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        self.write_beliefs()
        if self._trust == {}:
            self.initialize_trust()
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)

        if self._arrayWorld is None:
            self._arrayWorld = np.empty(state['World']['grid_shape'], dtype=list)

        self.update_info(receivedMessages)

        # Get agent location & close objects
        agentLocation = state[self.agent_id]['location']
        closeObjects = state.get_objects_in_area((agentLocation[0] - 1, agentLocation[1] - 1),
                                                 bottom_right=(agentLocation[0] + 1, agentLocation[1] + 1))
        # Filter out only blocks
        closeBlocks = None
        if closeObjects is not None:
            closeBlocks = [obj for obj in closeObjects
                           if 'CollectableBlock' in obj['class_inheritance']]

        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages, state, closeBlocks)

        # Update arrayWorld
        for obj in closeObjects:
            loc = obj['location']
            self._arrayWorld[loc[0]][loc[1]] = []

        if Phase.MOVE_TO_OBJECT or not Phase.DROP_OBJECT or not Phase.PLAN_PATH_TO_OBJECT or not Phase.PLAN_PATH_TO_OBJECT or not Phase.GRAB:
                self.found_next_obj()

        if self.__is_lazy() and self._phase != Phase.START and self._can_be_lazy and self._phase in [
            Phase.SEARCH_ROOM,
            Phase.MOVING_BLOCK,
            # Phase.MOVE_TO_OBJECT,
            Phase.SEARCH_RANDOM_ROOM,
            # Phase.PLAN_PATH_TO_CLOSED_DOOR
        ]:
            self._can_be_lazy = False
            self._phase = random.choice(
                [Phase.SEARCH_RANDOM_ROOM, Phase.PLAN_PATH_TO_OBJECT, Phase.PLAN_PATH_TO_CLOSED_DOOR])
            if self._phase == Phase.PLAN_PATH_TO_OBJECT and not self._current_obj:
                self._phase = random.choice(
                    [Phase.SEARCH_RANDOM_ROOM, Phase.PLAN_PATH_TO_CLOSED_DOOR])
            if state[agent_name]['is_carrying']:
                # self._goal_objects_found.remove(self._current_obj)
                obj = self._current_obj
                obj['location'] = state[agent_name]['location']
                self._goal_objects_found.append(obj)
                #self._sendMessage(self._current_obj, state[agent_name]['location'], agent_name)
                self._sendMessage(Util.droppingBlockMessage(self._current_obj,state[agent_name]['location']), agent_name)
                return DropObject.__name__, {'object_id': obj['obj_id']}
        else:
            self._can_be_lazy = True

        while True:
            #print(self._phase)
            # if there is a goal block alreay found go to it

            if not self._goal_objects and self._phase is not Phase.START:
                self._phase = Phase.SEARCH_RANDOM_ROOM

            if Phase.START == self._phase:
                # 'Drop_off_0', 'Drop_off_0_1', 'Drop_off_0_2'
                # {'drop_zone_nr': 0, 'is_drop_zone': True, 'is_goal_block': False, 'is_collectable': False, 'name': 'Drop off 0', 'obj_id': 'Drop_off_0', 'location': (12, 21), 'is_movable': False, 'carried_by': [], 'is_traversable': True,
                # Collect_Block', 'Collect_Block_1', 'Collect_Block_2'
                # {'drop_zone_nr': 0, 'is_drop_zone': False, 'is_goal_block': True, 'is_collectable': False, 'name': 'Collect Block', 'obj_id': 'Collect_Block', 'location': (12, 23), 'is_movable': False, 'carried_by': [], 'is_traversable': True

                self._goal_objects.append(state['Collect_Block'])
                self._goal_objects.append(state['Collect_Block_1'])
                self._goal_objects.append(state['Collect_Block_2'])

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._current_obj = None
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                        'is_open']]
                if len(closedDoors) == 0:
                    self._phase = Phase.SEARCH_RANDOM_ROOM
                    return None, {}
                # Randomly pick a closed door
                self._door = random.choice(closedDoors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage(Util.moveToMessage(self._door['room_name']), agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:

                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}

                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:

                # self._can_be_lazy = True
                self._navigator.reset_full()
                contents = state.get_room_objects(self._door['room_name'])
                waypoints = []
                for c in contents:
                    if "wall" not in c['name']:
                        x, y = c["location"][0], c["location"][1]
                        waypoints.append((x, y))

                self._navigator.add_waypoints(waypoints)

                # Open door
                is_open = state.get_room_doors(self._door['room_name'])[0]['is_open']
                if not is_open:
                    self._sendMessage(Util.openingDoorMessage(self._door['room_name']), agent_name)
                    self._sendMessage(Util.searchingThroughMessage(self._door['room_name']), agent_name)
                    return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}
                self._phase = Phase.SEARCH_ROOM
            if Phase.SEARCH_ROOM == self._phase:
                # if self.__is_lazy() and self._can_be_lazy:
                #     self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                # else:
                #     self._can_be_lazy = True
                self._state_tracker.update(state)
                contents = state.get_room_objects(self._door['room_name'])
                for c in contents:
                    if "Block" in c['name'] and c not in self._objects:
                        self._objects.append(c)

                    # self.foundBlockMessage(c, agent_name)
                    for i in range(len(self._goal_objects)):
                        if c['visualization']['colour'] == self._goal_objects[i]['visualization']['colour'] and \
                                c['visualization']['shape'] == self._goal_objects[i]['visualization']['shape'] and \
                                c['visualization']['size'] == self._goal_objects[i]['visualization']['size'] and \
                                not c['is_goal_block'] and not c['is_drop_zone'] and not self.already_delivered(c):
                            if i == 0:
                                #print('found')
                                #TODO Message sending should look like this!
                                self._sendMessage(Util.foundGoalBlockMessage(c), agent_name)
                                #self.foundGoalBlockMessage(c, agent_name)
                                self._phase = Phase.PLAN_PATH_TO_OBJECT
                                self._current_obj = c
                                return None, {}
                            else:
                                self._goal_objects_found.append(c)


                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_PATH_TO_OBJECT == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._current_obj['location']])
                self._phase = Phase.MOVE_TO_OBJECT

            if Phase.MOVE_TO_OBJECT == self._phase:

                #print("moving to obj")
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.GRAB
                # self._can_be_lazy = True

            if Phase.GRAB == self._phase:
                #print("ggrabbing")
                self._sendMessage(Util.pickingUpBlockMessage(self._current_obj), agent_name)
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goal_objects[0]['location']])
                self._phase = Phase.MOVING_BLOCK
                return GrabObject.__name__, {'object_id': self._current_obj['obj_id']}

            if Phase.MOVING_BLOCK == self._phase:

                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                # print("moving to block")

                if action is not None:
                    # print("is not none moving block")
                    return action, {}

                self._phase = Phase.DROP_OBJECT

            if Phase.DROP_OBJECT == self._phase:
                if state[agent_name]['is_carrying']:
                    #print("dropppgbg")
                    self._goal_object_delivered.append(self._goal_objects[0])
                    self._goal_objects.remove(self._goal_objects[0])
                    self._can_be_lazy = True
                    self._sendMessage(Util.droppingBlockMessage(self._current_obj,state[agent_name]['location']), agent_name)

                    return DropObject.__name__, {'object_id': self._current_obj['obj_id']}

                if self._goal_objects and self._goal_objects_found:
                    found = self.found_next_obj()
                    if not found:
                        self._phase = Phase.SEARCH_RANDOM_ROOM
                else:
                    self._phase = Phase.SEARCH_RANDOM_ROOM

            if Phase.SEARCH_RANDOM_ROOM == self._phase:
                doors = [door for door in state.values()
                         if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                self._door = random.choice(doors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                #self._sendMessage('Moving to door of ' + self._door['room_name'], agent_name)
                self._sendMessage(Util.moveToMessage(self._door['room_name']),agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

    def  _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def update_info(self, receivedMessages):
        # TODO act according to trust values
        for member in self._teamMembers:
            for msg in receivedMessages[member]:
                block = {
                    'is_drop_zone': False,
                    'is_goal_block': False,
                    'is_collectable': True,
                    'name': 'some_block',
                    'obj_id': 'some_block',
                    'location': (0, 0),
                    'is_movable': True,
                    'carried_by': [],
                    'is_traversable': True,
                    'class_inheritance': ['CollectableBlock', 'EnvObject', 'object'],
                    'visualization': {'size': -1, 'shape': -1, 'colour': '#00000', 'depth': 80, 'opacity': 1.0}}

                if "Found goal block " in msg:
                    pattern = re.compile("{(.* ?)}")
                    vis = re.search(pattern, msg).group(0)

                    pattern2 = re.compile("\((.* ?)\)")
                    loc = re.search(pattern2, msg).group(0)
                    loc = loc.replace("(", "[")
                    loc = loc.replace(")", "]")
                    loc = json.loads(loc)
                    vis = json.loads(vis)
                    block['location'] = (loc[0], loc[1])
                    block['visualization'] = vis

                    there = False
                    for b in self._goal_objects_found:
                        if b['visualization']['shape'] == vis['shape'] and b['visualization'][
                            'colour'] == vis['colour'] and not self.already_delivered(block):
                            there = True
                    if not there:
                        self._goal_objects_found.append(block)

                    if self._arrayWorld[block['location'][0], block['location'][1]] is None:
                        self._arrayWorld[block['location'][0], block['location'][1]] = []
                    self._arrayWorld[block['location'][0], block['location'][1]].append({
                        "memberName": member,
                        "block": block['visualization'],
                        "action": "found",
                    })


                elif "Picking up goal block " in msg:

                    pattern = re.compile("{(.* ?)}")
                    vis = re.search(pattern, msg).group(0)

                    pattern2 = re.compile("\((.* ?)\)")
                    loc = re.search(pattern2, msg).group(0)
                    loc = loc.replace("(", "[")
                    loc = loc.replace(")", "]")
                    loc = json.loads(loc)
                    vis = json.loads(vis)

                    block['location'] = (loc[0], loc[1])
                    block['visualization'] = vis

                    for b in self._goal_objects_found:
                        if b['visualization']['shape'] == vis['shape'] and b['visualization'][
                            'colour'] == vis['colour'] and b['location'] == (loc[0], loc[1]):
                            self._goal_objects_found.remove(b)
                    # for b in self._goal_objects:
                    #     if b['visualization'] == vis:
                    #         self._goal_objects.remove(b)

                    if self._arrayWorld[block['location'][0], block['location'][1]] is None:
                        self._arrayWorld[block['location'][0], block['location'][1]] = []
                    self._arrayWorld[block['location'][0], block['location'][1]].append({
                        "memberName": member,
                        "block": block['visualization'],
                        "action": "pickup"
                    })


                elif "Dropped goal block " in msg:
                    pattern = re.compile("{(.* ?)}")
                    vis = re.search(pattern, msg).group(0)

                    pattern2 = re.compile("\((.* ?)\)")
                    loc = re.search(pattern2, msg).group(0)
                    loc = loc.replace("(", "[")
                    loc = loc.replace(")", "]")
                    loc = json.loads(loc)
                    vis = json.loads(vis)
                    # do stuff
                    block['visualization'] = vis
                    block['location'] = loc

                    # there = False
                    # for b in self._goal_objects_found:
                    #     if b['visualization'] == vis:
                    #         there = True
                    #         b['location'] = (loc[0], loc[1])
                    # if not there:
                    #     self._goal_objects_found.append(block)
                    #
                    obj = None
                    for b in self._goal_objects:
                        if b['visualization']['shape'] == vis['shape'] and b['visualization']['colour'] == vis[
                            'colour'] and b['location'] == (loc[0], loc[1]):
                            obj = b
                        elif b['visualization']['shape'] == vis['shape'] and b['visualization'][
                            'colour'] == vis['colour'] and not b['location'] == (loc[0], loc[1]):
                            self._goal_objects_found.append(block)
                    if obj is not None:
                        self._goal_objects.remove(obj)
                        self._goal_object_delivered.append(obj)

                    if self._current_obj:
                        if self._current_obj['visualization']['shape'] == vis['shape'] and \
                                self._current_obj['visualization']['colour'] == vis['colour']:
                            self._phase = Phase.DROP_OBJECT

                    if self._arrayWorld[block['location'][0], block['location'][1]] is None:
                        self._arrayWorld[block['location'][0], block['location'][1]] = []
                    self._arrayWorld[block['location'][0], block['location'][1]].append({
                        "memberName": member,
                        "block": block['visualization'],
                        "action": "drop"
                    })



    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def read_trust(self):
        # agentname_trust.csv
        file_name = self.agent_id + '_trust.csv'
        #fprint(file_name)
        if os.path.exists(file_name):
            with open(file_name, newline='\n') as file:
                reader = csv.reader(file, delimiter=',')
                for row in reader:
                    self._trust[row[0]] = {"pick-up": row[1], "drop-off": row[2], "found": row[3], "average": row[4],
                                           "rep": row[5]}
        else:
            f = open(file_name, 'x')
            f.close()

        #print(self._trust)
    def initialize_trust(self):
        team = self._teamMembers
        for member in team:
            self._trust[member] = {"pick-up": 0.5, "drop-off": 0.5, "found": 0.5, "average": 0.5,
                                   "rep": 0.5}

    def write_beliefs(self):
        file_name = self.agent_id + '_trust.csv'
        with open(file_name, 'w') as file:
            # TODO add name to file
            writer = csv.DictWriter(file, ["pick-up", "drop-off", "found", "average", "rep"])
            #writer.writeheader()
            names = self._trust.keys()
            for name in names:
                writer.writerow(self._trust[name])


    def _trustBlief(self, member, received, state, close_objects):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference

        # Go throug the seen objects
        #print(self._arrayWorld)
        if close_objects is not None:
            for o in close_objects:
                loc = o['location']
                messages = self._arrayWorld[loc[0], loc[1]]
                # If we find messages for the location of the object
                if messages is not None and len(messages) > 0:
                    # If last message is 'pick-up' substract from trust
                    if messages[-1]['action'] == "pick-up":
                        member = messages[-1]['memberName']
                        self._trust[member]['pick-up'] = max(self._trust[member]['pick-up'] - 0.1, 0)
                    # If last message is 'found' or 'drop-of' add to trust
                    if messages[-1]['action'] == "found" or messages[-1]['drop-off'] == "found":
                        if o['visualization'] == messages[-1]['block']:
                            self._trust[member]['found'] = min(self._trust[member]['found'] + 0.1, 1)
                    if len(messages) > 1:
                        i = len(messages) - 2
                        while i >= 0:
                            member = messages[i]['memberName']
                            if messages[-1]['action'] == "drop-off":
                                self._trust[member]['drop-off'] = min(self._trust[member]['drop-off'] + 0.1, 1)
                                break
                            if not messages[-1]['action'] == "found":
                                break
                            if o['visualization'] == messages[-1]['block']:
                                self._trust[member]['found'] = min(self._trust[member]['found'] + 0.1, 1)
                            else:
                                self._trust[member]['found'] = max(self._trust[member]['found'] - 0.1, 0)
                            i -= 1


        agentLocation = state[self.agent_id]['location']
        for x in range(agentLocation[0] - 1, agentLocation[0] + 2):
            for y in range(agentLocation[1] - 1, agentLocation[0] + 2):
                messages = self._arrayWorld[x][y]
                if messages is not None and len(messages) > 0:
                    member = messages[-1]['memberName']
                    if isinstance(messages, list) and messages[-1]['action'] == "found" or messages[-1]['action'] == "drop-off":
                        if close_objects is None:
                            self._trust[member][messages[-1]['action']] = max(self._trust[member][messages[-1]['action']] - 0.1, 0)
                        else:
                            found = False
                            for o in close_objects:
                                if o['location'] == (x,y):
                                    if o['visualization'] == messages[-1]['block']:
                                        found = True
                            if found is False:
                                self._trust[member][messages[-1]['action']] = max(self._trust[member][messages[-1]['action']] - 0.1, 0)




    def __is_lazy(self):
        return random.randint(0, 1) == 1

    def already_delivered(self, o1):
        for o in self._goal_object_delivered:
            if o['visualization']['shape'] == o1['visualization']['shape'] and o['visualization']['colour'] == \
                    o1['visualization']['colour']:
                return True
        return False

    def found_next_obj(self):
        found = False
        for o in self._goal_objects_found:
            if o['visualization']['colour'] == self._goal_objects[0]['visualization']['colour'] and \
                    o['visualization']['shape'] == self._goal_objects[0]['visualization'][
                'shape'] and o not in self._goal_object_delivered:
                self._goal_objects_found.remove(o)
                self._phase = Phase.PLAN_PATH_TO_OBJECT
                self._current_obj = o
                found = True
                #print('moving to found one')
        return found
