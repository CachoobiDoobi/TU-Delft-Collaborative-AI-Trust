import csv
import enum
import os
import random
from typing import Dict

import numpy as np
from matrx.actions import GrabObject, DropObject
from matrx.actions.door_actions import OpenDoorAction
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message
from agents1.Util import Util
from bw4t.BW4TBrain import BW4TBrain


class Phase(enum.Enum):
    DROP_BLOCK = 9
    FOLLOW_PATH_TO_GOAL = 8
    PICKUP_BLOCK = 7
    FOUND_BLOCK = 6
    SEARCH_BLOCK = 5
    PREPARE_ROOM = 4
    PLAN_PATH_TO_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3


class StrongAgentRefactored(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._currentIndex = 0
        self._foundGoalBlocks = None
        self._currentRoomObjects = None
        self._door = None
        self._phase = Phase.PLAN_PATH_TO_DOOR
        self._teamMembers = []
        self._goalBlocks = []
        self._trust = {}
        self._arrayWorld = None
        self._holdingBlocks = []
        self._doors = None
        self._doorIndex = 0
        self._blockToPick = None

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
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
        # Update trust beliefs for team members
        # -----------------TRUST-----------------
        if self._trust == {}:
            self.initialize_trust()
            self.read_trust()
        self.write_beliefs()
        self._sendMessage(Util.reputationMessage(self._trust, self._teamMembers), agent_name)
        print(self._trust)
        # ------------------------------------
        self._prepareArrayWorld(state)
        self.updateGoalBlocks(state)
        self._holdingBlocks = state.get_self()['is_carrying']
        self._prepareDoors(state)

        Util.update_info_general(self._arrayWorld, receivedMessages, self._teamMembers,
                                 self.foundGoalBlockUpdate, self.foundBlockUpdate,
                                 self.pickUpBlockUpdate, self.dropBlockUpdate, self.dropGoalBlockUpdate, self.updateRep)

        self._updateWorld(state)

        while True:
            self.checkNextDropPossibility()
            if self.checkCurrentBlockDrop():
                self._phase = Phase.FOLLOW_PATH_TO_GOAL
            if Phase.PLAN_PATH_TO_DOOR == self._phase:
                self.PLAN_PATH_TO_DOOR_logic(agent_name)
            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._navigator.reset_full()
                self._phase = Phase.PREPARE_ROOM
                # Open door
                self._sendMessage(Util.openingDoorMessage(self._door['room_name']), agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}
            if Phase.PREPARE_ROOM == self._phase:
                self.PREPARE_ROOM_logic(state, agent_name)
            if Phase.SEARCH_BLOCK == self._phase:
                self._state_tracker.update(state)
                contents = state.get_room_objects(self._door['room_name'])
                for c in contents:
                    if ("Block" in c['name']) and (c not in self._currentRoomObjects) \
                            and 'GhostBlock' not in c['class_inheritance']:
                        self._currentRoomObjects.append(c)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.FOUND_BLOCK
            if Phase.FOUND_BLOCK == self._phase:
                self.FOUND_BLOCK_logic(agent_name)
            if Phase.PICKUP_BLOCK == self._phase:
                return self.pickupLogic(agent_name, self._blockToPick, state)
            if Phase.FOLLOW_PATH_TO_GOAL == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goalBlocks[self._currentIndex]['location']])
                self._phase = Phase.DROP_BLOCK
            if Phase.DROP_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                if len(self._holdingBlocks) > 0:
                    self._currentIndex += 1
                    block = self._holdingBlocks[-1]
                    self._sendMessage(Util.droppingBlockMessage(
                        block, self._goalBlocks[self._currentIndex - 1]['location']), agent_name)
                    return DropObject.__name__, {
                        'object_id': block['obj_id']}
            self._phase = Phase.PLAN_PATH_TO_DOOR
####################################################################################
########################### Action Logic ###########################################
    def FOUND_BLOCK_logic(self,agent_name):
        for c in self._currentRoomObjects:
            if self.isGoalBlock(c):
                self._sendMessage(Util.foundGoalBlockMessage(c), agent_name)
                self.manageBlock(c)
            else:
                self._sendMessage(Util.foundBlockMessage(c), agent_name)
        action = self._navigator.get_move_action(self._state_tracker)
        if action is not None:
            return action, {}
        self._phase = Phase.PLAN_PATH_TO_DOOR

    def PREPARE_ROOM_logic(self, state, agent_name):
        self._navigator.reset_full()
        contents = state.get_room_objects(self._door['room_name'])
        waypoints = []
        for c in contents:
            if 'class_inheritance' in c and 'AreaTile' in c['class_inheritance']:
                x, y = c["location"][0], c["location"][1]
                waypoints.append((x, y))

        self._navigator.add_waypoints(waypoints)
        self._currentRoomObjects = []
        self._phase = Phase.SEARCH_BLOCK
        self._sendMessage(Util.searchingThroughMessage(self._door['room_name']), agent_name)
    def PLAN_PATH_TO_DOOR_logic(self, agent_name):
        self._navigator.reset_full()
        if self._doorIndex >= len(self._doors):
            # Randomly pick a closed door
            self._door = random.choice(self._doors)
        else:
            self._door = self._doors[self._doorIndex]
            self._doorIndex += 1
        doorLoc = self._door['location']
        # Location in front of door is south from door
        doorLoc = doorLoc[0], doorLoc[1] + 1
        # Send message of current action
        self._sendMessage(Util.moveToMessage(self._door['room_name']), agent_name)
        self._navigator.add_waypoints([doorLoc])
        self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

    def pickupLogic(self, agent_name, currentBlock, state):
        self._state_tracker.update(state)
        action = self._navigator.get_move_action(self._state_tracker)
        if action is not None:
            return action, {}
        self._phase = Phase.PLAN_PATH_TO_DOOR
        goalBlockIndex = self.getGoalBlockIndex(currentBlock)
        if goalBlockIndex == None:
            return
        self._sendMessage(Util.pickingUpBlockMessage(currentBlock), agent_name)
        return GrabObject.__name__, {'object_id': currentBlock['obj_id']}
################################################################################
####################### Block Logic ############################################
    def checkNextDropPossibility(self):
        for i in range (self._currentIndex, len(self._goalBlocks)):
            if self._foundGoalBlocks[i] is not None and \
                i not in [self.getGoalBlockIndex(x) for x in self._holdingBlocks]:
                    self.manageBlock(self._foundGoalBlocks[i])
    def checkCurrentBlockDrop(self):
        if self._holdingBlocks is None or len(self._holdingBlocks) == 0:
            return False
        targetBlock = self._holdingBlocks[-1]
        goalBlockIndex = self.getGoalBlockIndex(targetBlock)
        if goalBlockIndex == self._currentIndex:
            return True
        return False
    def getGoalBlockIndex(self, block):
        getBlockInfo = lambda x: dict(list(x['visualization'].items())[:3])
        blockInfo = getBlockInfo(block)
        reducedGoalBlocks = [getBlockInfo(x) for x in self._goalBlocks]
        try:
            return reducedGoalBlocks.index(blockInfo)
        except ValueError:
            return None
    def updateGoalBlocks(self, state):
        if len(self._goalBlocks) == 0:
            self._goalBlocks = [goal for goal in state.values()
                                if 'is_goal_block' in goal and goal['is_goal_block']]
            self._foundGoalBlocks = np.empty(len(self._goalBlocks), dtype=dict)
    def isGoalBlock(self, block):
        getBlockInfo = lambda x: dict(list(x['visualization'].items())[:3])
        blockInfo = getBlockInfo(block)
        reducedGoalBlocks = [getBlockInfo(x) for x in self._goalBlocks]
        if (blockInfo in reducedGoalBlocks) and not block['is_goal_block'] and not block['is_drop_zone']:
            return True
        return False

    def manageBlock(self, block):
        goalBlockIndex = self.getGoalBlockIndex(block)
        self._foundGoalBlocks[goalBlockIndex] = block
        if goalBlockIndex in [self.getGoalBlockIndex(x) for x in self._holdingBlocks]:
            return
        if goalBlockIndex == self._currentIndex or (len(self._holdingBlocks) == 0 and goalBlockIndex > self._currentIndex):
            self._phase = Phase.PICKUP_BLOCK
            self._blockToPick = block
            self._navigator.reset_full()
            self._navigator.add_waypoints([block['location']])
########################################################################################################
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

    def _prepareArrayWorld(self, state):
        worldShape = state['World']['grid_shape']
        if self._arrayWorld is None:
            self._arrayWorld = np.empty(worldShape, dtype=list)
            for x in range(worldShape[0]):
                for y in range(worldShape[1]):
                    self._arrayWorld[x, y] = []

    def _prepareDoors(self, state):
        if self._doors is not None:
            return
        self._doors = []
        rooms = state.get_all_room_names()
        rooms.sort()
        print(rooms)
        #rooms = ['room_1',  'room_3', 'room_0',  'room_2', 'room_4', 'room_5', 'room_6', 'room_7', 'room_8']
        for room in rooms:
            currentDoor = state.get_room_doors(room)
            if len(currentDoor) > 0:
                self._doors.append(state.get_room_doors(room)[0])

    def _updateWorld(self, state):
        agentLocation = state[self.agent_id]['location']
        closeObjects = state.get_objects_in_area((agentLocation[0] - 2, agentLocation[1] - 2),
                                                 bottom_right=(agentLocation[0] + 2, agentLocation[1] + 2))
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

    ###################### TRUST ################################

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

        # print(self._trust)

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
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference

        # Go throug the seen objects
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

    ########################################################################
    ################# Update Info From Team ################################
    def foundGoalBlockUpdate(self, block, member):
        if self._trust[member]['found'] < 0.7:
            return
        goalBlockIndex = self.getGoalBlockIndex(block)
        if goalBlockIndex is None:
            return
        self._foundGoalBlocks[goalBlockIndex] = block

    def foundBlockUpdate(self, block, member):
        return

    def pickUpBlockUpdate(self, block, member):
        if (self._trust[member]['pick-up'] < 0.7 or self._trust[member]['verified'] < 3) and self._trust[member]['rep'] < 0.7:
            return
        goalBlockIndex = self.getGoalBlockIndex(block)
        if goalBlockIndex is None:
            return
        self._foundGoalBlocks[goalBlockIndex] = None

    def dropBlockUpdate(self, block, member):
        return

    def dropGoalBlockUpdate(self, block, member):
        if self._trust[member]['drop-off'] < 0.7:
            return
        goalBlockIndex = self.getGoalBlockIndex(block)
        if goalBlockIndex is None:
            return
        if self._goalBlocks[goalBlockIndex]['location'] == block['location']:
            if self._currentIndex == goalBlockIndex:
                self._currentIndex += 1
        else:
            self._foundGoalBlocks[goalBlockIndex] = block
    def updateRep(self, avg_reps):
        for member in self._teamMembers:
            self._trust[member]['rep'] = avg_reps[member] / len(self._teamMembers)


