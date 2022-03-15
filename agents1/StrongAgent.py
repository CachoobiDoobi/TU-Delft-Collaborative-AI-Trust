import enum
import random
from typing import Dict

from jsonpickle import json
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import DropObject
from matrx.actions.object_actions import GrabObject
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message

from bw4t.BW4TBrain import BW4TBrain
import numpy as np


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3
    SEARCH_BLOCK = 4
    FOUND_BLOCK = 5
    FOLLOW_PATH_TO_BLOCK = 6
    PICKUP_BLOCK = 7
    FOLLOW_PATH_TO_GOAL = 8
    DROP_BLOCK = 9
    PREPARE_ROOM = 10


class StrongAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        settings['max_objects'] = 2
        super().__init__(settings)
        self._door = None
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self._currentRoomObjects = []
        self._goalBlocks = []
        self._currentIndex = 0
        self._foundGoalBlocks = None
        self._holdingBlocks = []
    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

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
        self._trustBlief(self._teamMembers, receivedMessages)
        self.updateGoalBlocks(state)
        while True:
            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                        'is_open']]
                if len(closedDoors) == 0:
                    return None, {}
                # Randomly pick a closed door
                self._door = random.choice(closedDoors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self.moveToMessage(agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._navigator.reset_full()
                self._phase = Phase.PREPARE_ROOM
                # Open door
                self.openingDoorMessage(agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}
            if Phase.PREPARE_ROOM == self._phase:
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
                self.searchingThroughMessage(agent_name)
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
                for c in self._currentRoomObjects:
                    if not self.isGoalBlock(c):
                        self.foundBlockMessage(c, agent_name)
                    if self.isGoalBlock(c):
                        self.foundGoalBlockMessage(c, agent_name)
                        # most of the block picking logic
                        self.manageBlock(c)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
            if Phase.PICKUP_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                #needs more logic !!!
                self._phase = Phase.FOLLOW_PATH_TO_GOAL
                self.pickingUpBlockMessage(self._foundGoalBlocks[self._currentIndex], agent_name)
                return GrabObject.__name__, {'object_id': self._foundGoalBlocks[self._currentIndex]['obj_id']}
            if Phase.FOLLOW_PATH_TO_GOAL == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goalBlocks[self._currentIndex]['location']])
                self._phase = Phase.DROP_BLOCK
            if Phase.DROP_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                if state[agent_name]['is_carrying']:
                    self._currentIndex += 1
                    self.droppingBlockMessage(self._foundGoalBlocks[self._currentIndex - 1],
                                              self._goalBlocks[self._currentIndex - 1]['location'], agent_name)
                    return DropObject.__name__, {'object_id': self._foundGoalBlocks[self._currentIndex - 1]['obj_id']}
                if self._foundGoalBlocks[self._currentIndex] is not None:
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._foundGoalBlocks[self._currentIndex]['location']])
                    self._phase = Phase.PICKUP_BLOCK
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference
        default = 0.5
        trustBeliefs = {}
        for member in received.keys():
            trustBeliefs[member] = default
        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    trustBeliefs[member] -= 0.1
                    break
        return trustBeliefs

    #####################
    # GOAL BLOCKS LOGIC #
    #####################
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

    def getGoalBlockIndex(self, block):
        getBlockInfo = lambda x: dict(list(x['visualization'].items())[:3])
        blockInfo = getBlockInfo(block)
        reducedGoalBlocks = [getBlockInfo(x) for x in self._goalBlocks]
        return reducedGoalBlocks.index(blockInfo)

    def manageBlock(self, block):
        goalBlockIndex = self.getGoalBlockIndex(block)
        self._foundGoalBlocks[goalBlockIndex] = block
        if goalBlockIndex == self._currentIndex:
            self._phase = Phase.PICKUP_BLOCK
            self._navigator.reset_full()
            self._navigator.add_waypoints([block['location']])
    def manageBlock2(self, block):
        goalBlockIndex = self.getGoalBlockIndex(block)
        self._foundGoalBlocks[goalBlockIndex] = block
        if goalBlockIndex == self._currentIndex:
            if len(self._holdingBlocks) == 0:
                return
            if len(self._holdingBlocks) == 1:
                self._holdingBlocks.append(block)
                self._phase = Phase.PICKUP_BLOCK
                self._navigator.reset_full()
                self._navigator.add_waypoints([block['location']])
        else:
            if len(self._holdingBlocks) == 0:
                self._holdingBlocks.append(block)
                self._phase = Phase.PICKUP_BLOCK
                self._navigator.reset_full()
                self._navigator.add_waypoints([block['location']])
            if len(self._holdingBlocks) == 1:
                return
    def pickupLogic(self, agent_name):
        block = self._holdingBlocks[-1]
        goalBlockIndex = self.getGoalBlockIndex(block)
        if goalBlockIndex == self._currentIndex:
            self._phase = Phase.FOLLOW_PATH_TO_GOAL
            self.pickingUpBlockMessage(block, agent_name)
            return GrabObject.__name__, {'object_id': block['obj_id']}
        else:
            if len(self._holdingBlocks) == 1:
                if self._foundGoalBlocks[self._currentIndex] is not None:
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._foundGoalBlocks[self._currentIndex]['location']])
                    self._phase = Phase.PICKUP_BLOCK
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
            self.pickingUpBlockMessage(block, agent_name)
            return GrabObject.__name__, {'object_id': block['obj_id']}
    #########################
    # MESSAGE SENDING LOGIC #
    #########################
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

    def moveToMessage(self, agent_name):
        self._sendMessage('Moving to ' + self._door['room_name'], agent_name)

    def openingDoorMessage(self, agent_name):
        self._sendMessage('Opening door of ' + self._door['room_name'], agent_name)

    def searchingThroughMessage(self, agent_name):
        self._sendMessage('Searching through ' + self._door['room_name'], agent_name)

    def foundGoalBlockMessage(self, data, agent_name):
        item_info = dict(list(data['visualization'].items())[:3])
        self._sendMessage(
            "Found goal block: " + json.dumps(item_info) +
            " at location (" + ', '.join([str(loc) for loc in data['location']]) + ")", agent_name)
    def foundBlockMessage(self, data, agent_name):
        item_info = dict(list(data['visualization'].items())[:3])
        self._sendMessage(
            "Found block: " + json.dumps(item_info) +
            " at location (" + ', '.join([str(loc) for loc in data['location']]) + ")", agent_name)
    def pickingUpBlockMessage(self, data, agent_name):
        item_info = dict(list(data['visualization'].items())[:3])
        self._sendMessage(
            "Picking up goal block: " + json.dumps(item_info) +
            " at location (" + ', '.join([str(loc) for loc in data['location']]) + ")", agent_name)

    def droppingBlockMessage(self, data, location, agent_name):
        item_info = dict(list(data['visualization'].items())[:3])
        self._sendMessage(
            "Droppped goal block: " + json.dumps(item_info) +
            " at location (" + ', '.join([str(loc) for loc in location]) + ")", agent_name)
    #################################################################################################
