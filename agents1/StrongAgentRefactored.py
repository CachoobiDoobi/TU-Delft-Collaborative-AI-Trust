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
    DROP_BLOCK = 9,
    FOLLOW_PATH_TO_GOAL = 8,
    PICKUP_BLOCK = 7,
    FOUND_BLOCK = 6,
    SEARCH_BLOCK = 5,
    PREPARE_ROOM = 4,
    PLAN_PATH_TO_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3,
    CHECK_BLIND_DROP = 10,
    PATH_BLIND_DROP = 11


class StrongAgentRefactored(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._checkBlind = (False, False)
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
        self.receivedMessagesIndex = 0
        self._drop_location_blind = None
        self._ticks = 0
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
        print(receivedMessages)
        # Update trust beliefs for team members
        # -----------------TRUST-----------------
        if self._trust == {}:
            self.initialize_trust()
            self.read_trust()
        self.write_beliefs()

        # ------------------------------------
        self._prepareArrayWorld(state)
        self.updateGoalBlocks(state)

        self._holdingBlocks = state.get_self()['is_carrying']
        self._prepareDoors(state)
        self._updateWorld(state)
        self._sendMessage(Util.reputationMessage(self._trust, self._teamMembers), agent_name)
        Util.update_info_general(self._arrayWorld, receivedMessages, self._teamMembers,
                                 self.foundGoalBlockUpdate, self.foundBlockUpdate,
                                 self.pickUpBlockUpdate, self.dropBlockUpdate, self.dropGoalBlockUpdate, self.updateRep, agent_name)


        self.setBlindData()
        self._ticks += 1
        while True:
            droppableBlock = self.dropOldGoalBlock()
            if droppableBlock is not None:
                self._sendMessage(Util.droppingBlockMessage(
                    droppableBlock, self._goalBlocks[self._currentIndex - 1]['location']), agent_name)
                return DropObject.__name__, {
                    'object_id': droppableBlock['obj_id']}
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
                self.FOUND_BLOCK_logic(agent_name, self._currentRoomObjects)
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
            if Phase.PATH_BLIND_DROP == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._drop_location_blind])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.CHECK_BLIND_DROP

            if Phase.CHECK_BLIND_DROP == self._phase:
                self._checkBlind = (self._checkBlind[0], False)
                agentLocation = state[self.agent_id]['location']
                closeObjects = state.get_objects_in_area((agentLocation[0], agentLocation[1]),
                                                         bottom_right=(agentLocation[0] + 1, agentLocation[1] + 1))
                self._phase = Phase.PLAN_PATH_TO_DOOR
                # Filter out only blocks
                closeBlocks = None
                if closeObjects is not None:
                    closeBlocks = [obj for obj in closeObjects
                                   if 'CollectableBlock' in obj['class_inheritance']]
                if len(closeBlocks) == 0:
                    return None, {}
                for c in closeBlocks:
                    if self.isGoalBlock(c):
                        self._sendMessage(Util.foundGoalBlockMessage(c), agent_name)
                        self.manageBlock(c)
                    else:
                        self._sendMessage(Util.foundBlockMessage(c), agent_name)
            self._phase = Phase.PLAN_PATH_TO_DOOR

    ####################################################################################
    ########################### Action Logic ###########################################

    def FOUND_BLOCK_logic(self, agent_name, currentRoomObjects):
        for c in currentRoomObjects:
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
        if self._checkBlind[1]:
            self._phase = Phase.PATH_BLIND_DROP
            return None, {}

        if self._doorIndex >= len(self._doors):
            # Randomly pick a closed door
            self._door = random.choice(self._doors)
            if self._checkBlind[0]:
                self._phase = Phase.PATH_BLIND_DROP
                return None, {}
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
        if goalBlockIndex is None:
            return None, {}
        block = self.getGoalBlockName(state, currentBlock)
        if block is None:
            self._foundGoalBlocks[goalBlockIndex] = None
            return None, {}
        self._sendMessage(Util.pickingUpBlockMessage(block), agent_name)
        return GrabObject.__name__, {'object_id': block['obj_id']}

    ################################################################################
    ####################### Block Logic ############################################
    def dropOldGoalBlock(self):
        for block in self._holdingBlocks:
            if self.getGoalBlockIndex(block) < self._currentIndex:
                return block
        return None

    def getGoalBlockName(self, state, block):
        if block is None:
            return None
        closeObjects = state.get_objects_in_area((block['location'][0] - 1, block['location'][1] - 1), bottom_right = (block['location'][0] + 1, block['location'][1] + 1))
        #closeObjects = state.get_closest_objects()
        closeBlocks = None
        if closeObjects is not None:
            closeBlocks = [obj for obj in closeObjects
                           if 'CollectableBlock' in obj['class_inheritance']]
        if closeObjects is None:
            return None
        for b in closeBlocks:
            if b['visualization']['colour'] == block['visualization']['colour'] and \
                                b['visualization']['shape'] == block['visualization']['shape'] and \
                                b['visualization']['size'] == block['visualization']['size']:
                return b
        return None
    def checkNextDropPossibility(self):
        for i in range(self._currentIndex, len(self._goalBlocks)):
            if self._foundGoalBlocks[i] is not None and \
                    i not in [self.getGoalBlockIndex(x) for x in self._holdingBlocks] and self._ticks % 5 == 0:
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
        if block is None:
            return None
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
    def setBlindData(self):
        if self._drop_location_blind is None:
            loc = self._goalBlocks[-1]['location']
            self._drop_location_blind = (loc[0], loc[1] - 1)
        if self._doorIndex % 2:
            self._checkBlind = (True, self._checkBlind[1])
        else:
            self._checkBlind = (False, self._checkBlind[1])
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
        if goalBlockIndex == self._currentIndex or (
                len(self._holdingBlocks) == 0 and goalBlockIndex > self._currentIndex):
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
        random.shuffle(rooms)
        #rooms = ['room_3', 'room_1', 'room_0', 'room_2', 'room_4', 'room_5', 'room_6', 'room_7', 'room_8']
        for room in rooms:
            currentDoor = state.get_room_doors(room)
            if len(currentDoor) > 0:
                self._doors.append(state.get_room_doors(room)[0])

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

    ###################### TRUST ################################

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


    def checkPickUpInteraction(self, interactions): # assume interactions are for the same type of block(same visualization)

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
            #inter['block']
            actionFreq[action] += 1
            if i == len(interactions) - 1 and action != 'pick-up':
                lastActionNotCorrect = True #wrong! decrease trust
            if action == 'drop-off':
                if i == len(interactions) - 1:
                    break
                if interactions[i + 1]['action'] == 'found':
                    continue # good! can be continued!
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
                if i == len(interactions) - 1: # correct case!
                    continue # increase trust
                elif interactions[i + 1]['action'] == 'drop-off':
                    continue # good! can be continued!
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
            self.increaseDecreaseTrust(members, True, realBlock) # increase trust of all agents
        else:
            self.increaseDecreaseTrust(members, False) #decrease trust
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

    ########################################################################
    ################# Update Info From Team ################################
    def badTrustCondition(self, member):
        if (self._trust[member]['average'] < 0.7 or self._trust[member]['verified'] < 3) \
                and self._trust[member]['rep'] < 0.7:
            return True
        return False

    def foundGoalBlockUpdate(self, block, member):
        if self.badTrustCondition(member):
            return
        goalBlockIndex = self.getGoalBlockIndex(block)
        if goalBlockIndex is None:
            return
        if self._foundGoalBlocks[goalBlockIndex] is None:
            self._foundGoalBlocks[goalBlockIndex] = block

    def foundBlockUpdate(self, block, member):
        return

    def pickUpBlockUpdate(self, block, member):
        if self.badTrustCondition(member):
            return
        goalBlockIndex = self.getGoalBlockIndex(block)
        if goalBlockIndex is None:
            return
        if tuple(block['location']) == self._foundGoalBlocks[goalBlockIndex]['location']:
            self._foundGoalBlocks[goalBlockIndex] = None

    def dropBlockUpdate(self, block, member):
        if self.badTrustCondition(member):
            return
        if tuple(block['location']) == self._drop_location_blind:
            self._checkBlind = (self._checkBlind[0], True)
        return

    def dropGoalBlockUpdate(self, block, member):
        if self.badTrustCondition(member):
            return
        goalBlockIndex = self.getGoalBlockIndex(block)
        if goalBlockIndex is None:
            return
        if self._goalBlocks[goalBlockIndex]['location'] == tuple(block['location']):
            if self._currentIndex == goalBlockIndex:
                self._currentIndex += 1
                # TODO if hold current goal block drop it!
        else:
            self._foundGoalBlocks[goalBlockIndex] = block

    def updateRep(self, avg_reps):
        for member in self._teamMembers:
            self._trust[member]['rep'] = avg_reps[member] / len(self._teamMembers)
