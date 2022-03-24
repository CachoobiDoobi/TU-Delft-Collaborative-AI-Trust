import enum
import random
import os
import csv
from typing import Dict
import numpy as np

from matrx.actions.door_actions import OpenDoorAction
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message
from agents1.Util import Util

from bw4t.BW4TBrain import BW4TBrain


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_DOOR = 2,
    OPEN_DOOR = 3,
    PLAN_PATH_TO_UNSEARCHED_DOOR = 4,
    SEARCH_ROOM = 5,
    FIND_BLOCK = 6,
    GRAB = 7,
    MOVE_TO_OBJECT = 8,
    MOVING_BLOCK = 9


class LiarAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_UNSEARCHED_DOOR
        self._teamMembers = []
        self._goal_objects_found = []
        self._goal_objects = None
        self._goal_object_delivered = []
        self._current_obj = None
        self._objects = []
        self._door = None
        self._trust = {}
        self._arrayWorld = None
        self.read_trust()

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._searched_doors_index = 0

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
        # Update trust beliefs for team members

        if self._arrayWorld is None:
            self._arrayWorld = np.empty(state['World']['grid_shape'], dtype=list)

        Util.update_info_general(self._arrayWorld, receivedMessages, self._teamMembers,
                                 self.foundGoalBlockUpdate, self.foundBlockUpdate, self.pickUpBlockUpdate,
                                 self.dropBlockUpdate, self.dropGoalBlockUpdate)

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

        if self._goal_objects is None:
            self._goal_objects = [goal for goal in state.values()
                                  if 'is_goal_block' in goal and goal['is_goal_block']]

        while True:
            if Phase.PLAN_PATH_TO_UNSEARCHED_DOOR == self._phase:
                self._navigator.reset_full()
                # check each room in the given order
                doors = [door for door in state.values()
                         if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

                if self._searched_doors_index >= len(doors):
                    return None, {}
                self._door = doors[self._searched_doors_index]
                self._searched_doors_index += 1

                door_location = self._door['location']
                # Location in front of door is south from door
                door_location = door_location[0], door_location[1] + 1

                # Send message with a probability of 0.8 to lie
                lie = random.uniform(0, 1)
                if lie <= 0.2:
                    self._sendMessage(Util.moveToMessage(self._door['room_name']), agent_name)
                    # self.moveToMessage(agent_name)
                else:
                    self._sendMessage(Util.moveToMessageLie(self._door['room_name'], doors), agent_name)
                    # self.moveToMessageLie(agent_name, doors)
                self._navigator.add_waypoints([door_location])
                self._phase = Phase.FOLLOW_PATH_TO_DOOR

            if Phase.FOLLOW_PATH_TO_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}

                if not self._door['is_open']:
                    self._phase = Phase.OPEN_DOOR
                else:
                    self._phase = Phase.SEARCH_ROOM

            if Phase.OPEN_DOOR == self._phase:
                # Send message with a probability of 0.8 to lie
                lie = random.uniform(0, 1)
                if lie <= 0.2:
                    self._sendMessage(Util.openingDoorMessage(self._door['room_name']), agent_name)
                else:
                    self._sendMessage(Util.openingDoorMessageLie(state, self._door['room_name']), agent_name)

                self._phase = Phase.SEARCH_ROOM
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.SEARCH_ROOM == self._phase:
                self._navigator.reset_full()
                room_area = []
                for area in state.get_room_objects(self._door['room_name']):
                    if "wall" not in area['name']:
                        room_area.append((area["location"][0], area["location"][1]))

                lie = random.uniform(0, 1)
                if lie <= 0.2:
                    self._sendMessage(Util.searchingThroughMessage(self._door['room_name']), agent_name)
                    # self.searchingThroughMessage(agent_name)
                else:
                    self._sendMessage(Util.searchingThroughMessageLie(state, self._door['room_name']),agent_name)
                    # self.searchingThroughMessageLie(agent_name, state)

                self._navigator.add_waypoints(room_area)
                self._phase = Phase.FIND_BLOCK

            if Phase.FIND_BLOCK == self._phase:
                self._state_tracker.update(state)

                contents = state.get_room_objects(self._door['room_name'])
                for c in contents:
                    goal = False
                    for i in range(len(self._goal_objects)):
                        if c['visualization']['colour'] == self._goal_objects[i]['visualization']['colour'] and \
                                c['visualization']['shape'] == self._goal_objects[i]['visualization']['shape'] and \
                                c['visualization']['size'] == self._goal_objects[i]['visualization']['size'] and \
                                not c['is_goal_block'] and not c['is_drop_zone']:
                            if i == 0:
                                goal = True
                                if not self._objects.__contains__(c):
                                    lie = random.uniform(0, 1)
                                    if lie <= 0.2:
                                        self._sendMessage(Util.foundGoalBlockMessage(c), agent_name)
                                    else:
                                        self._sendMessage(Util.foundBlockMessageLie(), agent_name)
                                    self._objects.append(c)
                                self._phase = Phase.MOVE_TO_OBJECT
                                self._current_obj = c
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([c['location']])
                                action = self._navigator.get_move_action(self._state_tracker)
                                return action, {}
                            else:
                                self._goal_objects_found.append(c)

                    if "Block" in c['name']:
                        if not self._objects.__contains__(c):
                            lie = random.uniform(0, 1)
                            if lie <= 0.2:
                                self._sendMessage(Util.foundGoalBlockMessage(c), agent_name)
                            else:
                                self._sendMessage(Util.foundBlockMessageLie(), agent_name)
                            self._objects.append(c)


                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.PLAN_PATH_TO_UNSEARCHED_DOOR

            if Phase.MOVE_TO_OBJECT == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.GRAB
                return GrabObject.__name__, {'object_id': self._current_obj['obj_id']}

            if Phase.GRAB == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goal_objects[0]['location']])
                self._phase = Phase.MOVING_BLOCK
                lie = random.uniform(0, 1)
                if lie <= 0.2:
                    self._sendMessage(Util.pickingUpBlockMessage(self._current_obj), agent_name)
                else:
                    self._sendMessage(Util.pickingUpBlockMessageLie(), agent_name)

            if Phase.MOVING_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}

                if state[agent_name]['is_carrying']:
                    lie = random.uniform(0, 1)
                    if lie <= 0.2:
                        self._sendMessage(Util.droppingBlockMessage(self._current_obj, state[agent_name]['location']), agent_name)
                    else:
                        self._sendMessage(Util.droppingBlockMessageLie(), agent_name)
                    self._goal_objects.remove(self._goal_objects[0])
                    return DropObject.__name__, {'object_id': self._current_obj['obj_id']}

                if self._goal_objects and self._goal_objects_found:
                    for object in self._goal_objects_found:
                        if object['visualization']['colour'] == self._goal_objects[0]['visualization']['colour'] and \
                                object['visualization']['shape'] == self._goal_objects[0]['visualization']['shape']:
                            self._navigator.reset_full()
                            self._goal_objects_found.remove(object)
                            self._navigator.add_waypoints([object['location']])
                            self._phase = Phase.MOVE_TO_OBJECT
                            self._current_obj = object
                else:
                    self._phase = Phase.PLAN_PATH_TO_UNSEARCHED_DOOR

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

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
        file_name = str(self.agent_id) + '_trust.csv'
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
        file_name = str(self.agent_id) + '_trust.csv'
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
        # print(self._arrayWorld)
        print("zzz: ", self._trust)
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
                    if messages[-1]['action'] == "found" or messages[-1]['action'] == "drop-off":
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
                    if isinstance(messages, list) and messages[-1]['action'] == "found" or messages[-1][
                        'action'] == "drop-off":
                        if close_objects is None:
                            self._trust[member][messages[-1]['action']] = max(
                                self._trust[member][messages[-1]['action']] - 0.1, 0)
                        else:
                            found = False
                            for o in close_objects:
                                if o['location'] == (x, y):
                                    if o['visualization'] == messages[-1]['block']:
                                        found = True
                            if found is False:
                                self._trust[member][messages[-1]['action']] = max(
                                    self._trust[member][messages[-1]['action']] - 0.1, 0)


    def foundGoalBlockUpdate(self, block, member):
        return

    def pickUpBlockUpdate(self, block, member):
        return

    def foundBlockUpdate(self, block, member):
        return

    def dropBlockUpdate(self, block, member):
        return

    def dropGoalBlockUpdate(self, block, member):
        return

    def updateGoalBlocks(self, state):
        return
