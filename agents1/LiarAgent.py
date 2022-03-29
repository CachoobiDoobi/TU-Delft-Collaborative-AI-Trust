import enum
import json
import random
import os
import csv
import re
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
    MOVING_BLOCK = 9,
    CHECK_GOAL_ZONE = 10,
    CHOOSE_NEXT_MOVE = 11


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

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._searched_doors_index = 0
        self.read_trust()

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        agent_name = state[self.agent_id]['obj_id']

        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)

        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)

        # Write trust
        if self._trust == {}:
            self.initialize_trust()
            self.read_trust()
        self.write_beliefs()
        # self._sendMessage(Util.reputationMessage(self._trust, self._teamMembers), agent_name)

        # Update information based on gathered information
        if self._arrayWorld is None:
            self._arrayWorld = np.empty(state['World']['grid_shape'], dtype=list)

        Util.update_info_general(self._arrayWorld, receivedMessages, self._teamMembers,
                                 self.foundGoalBlockUpdate, self.foundBlockUpdate, self.pickUpBlockUpdate,
                                 self.dropBlockUpdate, self.dropGoalBlockUpdate, self.updateRep)

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
        self._trustBlief(state, closeBlocks)

        # Update arrayWorld
        for obj in closeObjects:
            loc = obj['location']
            self._arrayWorld[loc[0]][loc[1]] = []

        if self._goal_objects is None:
            self._goal_objects = [goal for goal in state.values()
                                  if 'is_goal_block' in goal and goal['is_goal_block']]

        while True:
            # print(self._phase)

            if Phase.PLAN_PATH_TO_UNSEARCHED_DOOR == self._phase:
                self._navigator.reset_full()
                # check each room in the given order
                doors = [door for door in state.values()
                         if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

                if self._searched_doors_index >= len(doors):
                    check_next_to_goal_zone = random.uniform(0, 1)
                    if check_next_to_goal_zone <= 0.5:
                        self._phase = Phase.CHECK_GOAL_ZONE
                        return None, {}
                    else:
                        self._door = random.choice(doors)
                #     self._phase = Phase.CHECK_GOAL_ZONE
                #     print("a dat val")
                #     return None, {}
                else:
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
                    if "wall" not in area['name'] and not self._teamMembers.__contains__(area['name']):
                        room_area.append((area["location"][0], area["location"][1]))

                lie = random.uniform(0, 1)
                if lie <= 0.2:
                    self._sendMessage(Util.searchingThroughMessage(self._door['room_name']), agent_name)
                    # self.searchingThroughMessage(agent_name)
                else:
                    self._sendMessage(Util.searchingThroughMessageLie(state, self._door['room_name']), agent_name)
                    # self.searchingThroughMessageLie(agent_name, state)

                self._navigator.add_waypoints(room_area)
                self._phase = Phase.FIND_BLOCK

            if Phase.FIND_BLOCK == self._phase:
                self._state_tracker.update(state)

                contents = state.get_room_objects(self._door['room_name'])
                for c in contents:
                    # goal = False
                    for i in range(len(self._goal_objects)):
                        if c['visualization']['colour'] == self._goal_objects[i]['visualization']['colour'] and \
                                c['visualization']['shape'] == self._goal_objects[i]['visualization']['shape'] and \
                                c['visualization']['size'] == self._goal_objects[i]['visualization']['size'] and \
                                not c['is_goal_block'] and not c['is_drop_zone']:
                            if i == 0:
                                # goal = True
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
                if not state[agent_name]['is_carrying']:
                    self._phase = Phase.PLAN_PATH_TO_UNSEARCHED_DOOR
                    return None, {}
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

                # as long as it can move the block
                if action is not None:
                    return action, {}

                if state[agent_name]['is_carrying']:
                    lie = random.uniform(0, 1)
                    if lie <= 0.2:
                        self._sendMessage(Util.droppingBlockMessage(self._current_obj, state[agent_name]['location']),
                                          agent_name)
                    else:
                        self._sendMessage(Util.droppingBlockMessageLie(), agent_name)
                    self._goal_objects.remove(self._goal_objects[0])
                    self._phase = Phase.CHOOSE_NEXT_MOVE
                    return DropObject.__name__, {'object_id': self._current_obj['obj_id']}
                else:
                    self._phase = Phase.CHOOSE_NEXT_MOVE

            if Phase.CHOOSE_NEXT_MOVE == self._phase:
                # print("aici!!!!!")
                self._phase = Phase.PLAN_PATH_TO_UNSEARCHED_DOOR
                # if there is an goal object that we have previously found
                if len(self._goal_objects_found) > 0 and len(self._goal_objects) > 0:

                    # print("in if")
                    for obj in self._goal_objects_found:
                        if obj['visualization']['colour'] == self._goal_objects[0]['visualization']['colour'] and \
                                obj['visualization']['shape'] == self._goal_objects[0]['visualization']['shape']:
                            (picked, location) = \
                                self.checkLocationOfBlock(receivedMessages, obj['location'],
                                                          obj['visualization'])
                            if picked is False:
                                location = [obj['location']]
                            elif picked is True and location is None:

                                self._phase = Phase.PLAN_PATH_TO_UNSEARCHED_DOOR
                                break

                            self._navigator.reset_full()
                            self._goal_objects_found.remove(obj)
                            if isinstance(location, tuple):
                                self._navigator.add_waypoints([location])
                            else:
                                self._navigator.add_waypoints(location)
                            self._phase = Phase.MOVE_TO_OBJECT
                            self._current_obj = obj

            if Phase.CHECK_GOAL_ZONE == self._phase:
                loc = [goal for goal in state.values()
                       if 'is_goal_block' in goal and goal['is_goal_block']][2]['location']
                goal_zone = (loc[0], loc[1] - 1)
                self._navigator.add_waypoints([goal_zone])
                # self._phase = Phase.FOLLOW_PATH_TO_DOOR
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                objects = state.get_objects_in_area((goal_zone[0], goal_zone[1]),
                                                    bottom_right=(goal_zone[0], goal_zone[1]))
                # Filter out only blocks
                closeBlocks = None
                if objects is not None:
                    closeBlocks = [obj for obj in closeObjects
                                   if 'CollectableBlock' in obj['class_inheritance']]

                for c in closeBlocks:
                    for i in range(len(self._goal_objects)):
                        if c['visualization']['colour'] == self._goal_objects[i]['visualization']['colour'] and \
                                c['visualization']['shape'] == self._goal_objects[i]['visualization']['shape'] and \
                                c['visualization']['size'] == self._goal_objects[i]['visualization']['size'] and \
                                not c['is_goal_block'] and not c['is_drop_zone']:
                            if i == 0:
                                self._phase = Phase.MOVE_TO_OBJECT
                                self._current_obj = c
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([c['location']])
                                action = self._navigator.get_move_action(self._state_tracker)
                                return action, {}
                            else:
                                self._goal_objects_found.append(c)

                action = self._navigator.get_move_action(self._state_tracker)
                # print("a fct asta aici")
                if action is not None:
                    return action, {}
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

    ###############
    #    TRUST    #
    ###############

    def read_trust(self):
        # agentname_trust.csv
        file_name = self.agent_id + '_trust.csv'
        # fprint(file_name)
        if os.path.exists(file_name):
            with open(file_name, newline='') as file:
                reader = csv.reader(file, delimiter=',')
                for row in reader:
                    if row:
                        self._trust[row[0]] = {"pick-up": float(row[1]), "drop-off": float(row[2]),
                                               "found": float(row[3]),
                                               "average": float(row[4]),
                                               "rep": float(row[5]), "verified": float(row[6])}
        else:
            f = open(file_name, 'x')
            f.close()

    def initialize_trust(self):
        team = self._teamMembers
        for member in team:
            self._trust[member] = {"name": member, "pick-up": 0.5, "drop-off": 0.5, "found": 0.5, "average": 0.5,
                                   "rep": 0.5, "verified": 0}

    def write_beliefs(self):
        file_name = self.agent_id + '_trust.csv'
        with open(file_name, 'w') as file:
            writer = csv.DictWriter(file, ["name", "pick-up", "drop-off", "found", "average", "rep", "verified"])
            # writer.writeheader()
            names = self._trust.keys()
            for name in names:
                row = self._trust[name]
                row['name'] = name
                writer.writerow(row)

    def _trustBlief(self, state, close_objects):
        # print(self._arrayWorld)
        # print("l: ", self._trust)
        if close_objects is not None:
            for o in close_objects:
                loc = o['location']
                messages = self._arrayWorld[loc[0], loc[1]]
                # If we find messages for the location of the object
                if messages is not None and len(messages) > 0:
                    member = messages[-1]['memberName']
                    # If last message is 'pick-up' substract from trust
                    if messages[-1]['action'] == "pick-up":
                        self._trust[member]['pick-up'] = max(round(self._trust[member]['pick-up'] - 0.1, 3), 0)
                    # If last message is 'found' or 'drop-of' add to trust
                    if messages[-1]['action'] == "found" or messages[-1]['action'] == "drop-off":
                        val = self.check_same_visualizations(o['visualization'], messages[-1]['block'])
                        self._trust[member]['found'] = min(round(self._trust[member]['found'] + val, 3), 1)
                    if len(messages) > 1:
                        i = len(messages) - 2
                        while i >= 0:
                            member = messages[i]['memberName']
                            if messages[-1]['action'] == "drop-off":
                                self._trust[member]['drop-off'] = min(round(self._trust[member]['drop-off'] + 0.1, 3),
                                                                      1)
                                break
                            if not messages[-1]['action'] == "found":
                                break

                            val = self.check_same_visualizations(o['visualization'], messages[-1]['block'])
                            self._trust[member]['found'] = min(round(self._trust[member]['found'] + val, 3), 1)

                            i -= 1

        agentLocation = state[self.agent_id]['location']
        for x in range(agentLocation[0] - 2, agentLocation[0] + 2):
            for y in range(agentLocation[1] - 2, agentLocation[1] + 2):
                messages = self._arrayWorld[x][y]
                if messages is not None and len(messages) > 0:
                    member = messages[-1]['memberName']
                    if isinstance(messages, list) and messages[-1]['action'] == "found" or messages[-1][
                        'action'] == "drop-off":
                        if close_objects is None:
                            self._trust[member][messages[-1]['action']] = max(
                                round(self._trust[member][messages[-1]['action']] - 0.1, 3), 0)
                        else:
                            found = False
                            for o in close_objects:
                                if o['location'] == (x, y):
                                    found = True
                            if found is False:
                                self._trust[member][messages[-1]['action']] = max(
                                    round(self._trust[member][messages[-1]['action']] - 0.1, 3), 0)

    def check_same_visualizations(self, vis1, vis2):
        size = 0
        shape = 0
        colour = 0

        if "size" in vis1 and "size" in vis2:
            size = 0.033 if vis1['size'] == vis2['size'] else -0.033

        if "shape" in vis1 and "shape" in vis2:
            shape = 0.033 if vis1['shape'] == vis2['shape'] else -0.033

        if "colour" in vis1 and "colour" in vis2:
            colour = 0.033 if vis1['colour'] == vis2['colour'] else -0.033

        return size + shape + colour

    ############################
    #     CHECK GOAL BLOCK     #
    ############################

    def checkLocationOfBlock(self, receivedMessages, knownLocation, visualzation):
        found = False
        location = None
        for member in self._teamMembers:
            for message in receivedMessages[member]:
                # print("nope: ", member, ": ", str(message))
                # print("pppp: ", str(knownLocation))
                if str(message).startswith("Picking up ") and str(message).endswith(str(knownLocation)):
                    shape = "\"shape\": " + str(visualzation['shape'])
                    # print("l-a luat")
                    for m in reversed(receivedMessages[member]):
                        if str(m).startswith("Dropped ") and str(m).__contains__(shape):
                            # print("l-a si lasat")
                            pattern = re.compile("{(.* ?)}")
                            vis = re.search(pattern, m).group(0)

                            pattern2 = re.compile("\((.* ?)\)")
                            loc = re.search(pattern2, m).group(0)
                            loc = loc.replace("(", "[")
                            loc = loc.replace(")", "]")
                            loc = json.loads(loc)
                            vis = json.loads(vis)
                            location = (loc[0], loc[1])
                            break
                    # print("l-am gasit")
                    found = True
                    break

        return found, location

    ##########################
    #     UTIL FUNCTIONS     #
    ##########################

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

    def updateRep(self, avg):
        return
