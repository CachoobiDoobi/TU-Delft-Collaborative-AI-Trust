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
        self.receivedMessagesIndex = 0

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._searched_doors_index = None
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

        self._prepareArrayWorld(state)
        self.updateGoalBlocks(state)

        # Update information based on gathered information
        self._updateWorld(state)
        self._sendMessage(Util.reputationMessage(self._trust, self._teamMembers), agent_name)
        Util.update_info_general(self._arrayWorld, receivedMessages, self._teamMembers, self.foundGoalBlockUpdate,
                                 self.foundBlockUpdate, self.pickUpBlockUpdate, self.pickUpBlockSimpleUpdate, self.dropBlockUpdate,
                                 self.dropGoalBlockUpdate, self.updateRep, agent_name)

        # Get agent location & close objects
        agentLocation = state[self.agent_id]['location']
        closeObjects = state.get_objects_in_area((agentLocation[0] - 1, agentLocation[1] - 1),
                                                 bottom_right=(agentLocation[0] + 1, agentLocation[1] + 1))
        # Filter out only blocks
        closeBlocks = None
        if closeObjects is not None:
            closeBlocks = [obj for obj in closeObjects
                           if 'CollectableBlock' in obj['class_inheritance']]

        if self._searched_doors_index is None:
            self._searched_doors_index = list(range(0, len(state.get_all_room_names()) - 1))
        # Update trust beliefs for team members
        # self._trustBlief(state, closeBlocks)

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

                if len(self._searched_doors_index) <= 0:
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
                    print(self._searched_doors_index)
                    index = random.choice(self._searched_doors_index)
                    self._door = doors[index]
                    self._searched_doors_index.remove(index)

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
        reduced_received_messages = self.received_messages[self.receivedMessagesIndex:]
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in reduced_received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        self.receivedMessagesIndex = len(self.received_messages)
        return receivedMessages

    #######################
    #    PREPARE WORLD    #
    #######################

    def _prepareArrayWorld(self, state):
        worldShape = state['World']['grid_shape']
        if self._arrayWorld is None:
            self._arrayWorld = np.empty(worldShape, dtype=list)
            for x in range(worldShape[0]):
                for y in range(worldShape[1]):
                    self._arrayWorld[x, y] = []

    def _updateWorld(self, state):
        agentLocation = state[self.agent_id]['location']
        closeObjects = state.get_objects_in_area((agentLocation[0] - 1, agentLocation[1] - 1),
                                                 bottom_right=(agentLocation[0] + 1, agentLocation[1] + 1))
        # Filter out only blocks
        closeBlocks = None
        if closeObjects is not None:
            closeBlocks = [obj for obj in closeObjects
                           if 'CollectableBlock' in obj['class_inheritance']]

        # Update trust beliefs for team members
        self._trustBlief2(state, closeBlocks)

        for member in self._teamMembers:
            avg = 0
            for key in self._trust[member].keys():
                if key in ["pick-up", "drop-off", "found"]:
                    avg += self._trust[member][key] / 3.0
            self._trust[member]['average'] = avg

    ###############
    #    TRUST    #
    ###############

    def _trustBlief2(self, state, close_objects):
        agentLocation = state[self.agent_id]['location']
        (x, y) = agentLocation
        messages = self._arrayWorld[x][y]
        self._arrayWorld[x][y] = []
        if len(messages) > 0: #there is some sort of block interaction!
            realBlock = self.getObjectAtLocation(close_objects, (x, y))
            if realBlock == "MultipleObj":
                return
            if realBlock is None: #no actual block there so interaction must end with pickup to be valid!
                self.checkPickUpInteraction(messages)
            else: #block is there so interaction must end with found or drop-off to be valid!
                self.checkFoundInteraction(messages, realBlock)

    def checkPickUpInteraction(self,
                               interactions):  # assume interactions are for the same type of block(same visualization)

        actionFreq = {
            "drop-off": 0,
            "found": 0,
            "pick-up": 0
        }
        properActionOrder = True
        lastActionNotCorrect = False
        members = []
        for i in range(len(interactions)):
            inter = interactions[i]
            action = inter['action']
            members.append((inter['memberName'], action))
            # inter['block']
            actionFreq[action] += 1
            if i == len(interactions) - 1 and action != 'pick-up':
                lastActionNotCorrect = True  # wrong! decrease trust
            if action == 'drop-off':
                if i == len(interactions) - 1:
                    break
                if interactions[i + 1]['action'] == 'found':
                    continue  # good! can be continued!
                else:
                    properActionOrder = False
            elif action == 'found':
                if i == len(interactions) - 1:
                    break
                if interactions[i + 1]['action'] == 'found' or interactions[i + 1]['action'] == 'pick-up':
                    continue  # good! can be continued!
                else:
                    properActionOrder = False
            elif action == 'pick-up':
                if i == len(interactions) - 1:  # correct case!
                    continue  # increase trust
                elif interactions[i + 1]['action'] == 'drop-off':
                    continue  # good! can be continued!
                else:
                    properActionOrder = False
        if properActionOrder and not lastActionNotCorrect:
            if actionFreq["drop-off"] + actionFreq['found'] < 1 and actionFreq['pick-up'] == 1:
                self.increaseDecreaseTrust(members, False)  # decrease (cannot pickup block that has never been found!!)
            self.increaseDecreaseTrust(members, True)  # increase trust of all agents
        elif properActionOrder and lastActionNotCorrect:
            if actionFreq["drop-off"] + actionFreq['found'] > 1:
                return  # keep the same trust
            else:
                self.increaseDecreaseTrust(members, False)  # decrease trust
        else:
            self.increaseDecreaseTrust(members, False)  # decrease trust

    def increaseDecreaseTrust(self, members, isIncrease, block=None):

        val = -0.1
        if isIncrease:
            val = 0.1
        for member in members:
            if block is not None:
                val = self.check_same_visualizations(block['visualization'], member[2])
            self._trust[member[0]][member[1]] = min(max(round(self._trust[member[0]][member[1]] + val, 3), 0), 1)
            self._trust[member[0]]['verified'] += 1

    def checkFoundInteraction(self, interactions, realBlock):
        actionFreq = {
            "drop-off": 0,
            "found": 0,
            "pick-up": 0
        }
        properActionOrder = True
        lastActionNotCorrect = False
        members = []
        for i in range(len(interactions)):
            inter = interactions[i]
            action = inter['action']
            members.append((inter['memberName'], action, inter['block']))
            # inter['block']
            actionFreq[action] += 1
            if i == len(interactions) - 1:
                if action == 'pick-up':
                    lastActionNotCorrect = True  # wrong! decrease trust
                break
            if action == 'drop-off':
                if interactions[i + 1]['action'] == 'found':
                    continue  # good! can be continued!
                else:
                    properActionOrder = False
            elif action == 'found':
                if interactions[i + 1]['action'] == 'found' or interactions[i + 1]['action'] == 'pick-up':
                    continue  # good! can be continued!
                else:
                    properActionOrder = False
            elif action == 'pick-up':
                if interactions[i + 1]['action'] == 'drop-off':
                    continue  # good! can be continued!
                else:
                    properActionOrder = False

        if properActionOrder and not lastActionNotCorrect:
            self.increaseDecreaseTrust(members, True, realBlock)  # increase trust of all agents
        else:
            self.increaseDecreaseTrust(members, False)  # decrease trust

    def getObjectAtLocation(self, close_objects, location):
        closeBlocks = None
        if close_objects is not None:
            closeBlocks = [obj for obj in close_objects
                           if location == obj['location']]
        if len(closeBlocks) == 0:
            return None
        if len(closeBlocks) != 1:
            return "MultipleObj"
        return closeBlocks[0]
    def check_same_visualizations(self, vis1, vis2):
        size = 0
        shape = 0
        colour = 0
        if "shape" in vis1 and "shape" in vis2:
            shape = 0.05 if vis1['shape'] == vis2['shape'] else -0.05

        if "colour" in vis1 and "colour" in vis2:
            colour = 0.05 if vis1['colour'] == vis2['colour'] else -0.05

        return size + shape + colour

    ##########################
    #    READ WRITE TRUST    #
    ##########################

    def read_trust(self):
        # agentname_trust.csv
        file_name = self.agent_id + '_trust.csv'
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
            self._trust[member] = {"pick-up": 0.5, "drop-off": 0.5, "found": 0.5, "average": 0.5,
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
                if str(message).startswith("Picking up ") and str(message).endswith(str(knownLocation)) and\
                        self._trust[member]['average'] >= 0.7:
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

    def pickUpBlockSimpleUpdate(self, block, member):
        return

    def updateRep(self, avg_reps):
        nr_team_mates = len(self._teamMembers)
        for member in avg_reps.keys():
            self._trust[member]['rep'] = (avg_reps[member] + (
                        self._trust[member]['rep'] * (nr_team_mates - 1))) / nr_team_mates
