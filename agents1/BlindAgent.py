import enum
import random
import json
import csv
import os
import numpy as np
from typing import Dict

from matrx.actions.door_actions import OpenDoorAction
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message
from matrx.actions.object_actions import GrabObject, DropObject
from bw4t.BW4TBrain import BW4TBrain
from agents1.Util import Util


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3,
    SEARCH_ROOM = 4,

    PICKUP_BLOCK = 5,
    MOVING_BLOCK = 6,
    DROP_BLOCK = 7,

    CHECK_GOAL_TO_FOLLOW = 8,
    FOLLOW_PATH_TO_GOAL_BLOCK = 9,
    GRAB = 10,

    SEARCH_RANDOM_ROOM = 11,
    CHECK_PICKUP_BY_OTHER = 12,
    GO_CHECK_PICKUP = 13,


class BlindAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._door = None
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

        self._objects = []                                  # All objects found
        self._goal_blocks = []                              # Universally known goal blocks
        self._goal_blocks_locations = []                    # Goal block locations as said by other agents
        self._goal_blocks_locations_followed = []           # Goal block locations already followed (by blind)
        self._goal_blocks_location_followed_by_others = []  # Goal block locations where other agents went (from messages)

        self._trustBeliefs = []

        self._current_obj = None
        self._drop_location_blind = None

        self._trust = {}
        self._arrayWorld = None
        self.read_trust()

    def filter_bw4t_observations(self, state):
        for obj in state:
            if "is_collectable" in state[obj] and state[obj]['is_collectable']:
                state[obj]['visualization'].pop('colour')
        return state

    def decide_on_bw4t_action(self, state: State):
        if self._trust == {}:
            self.initialize_trust()

        # TODO
        # self.write_beliefs()

        state = self.filter_bw4t_observations(state)
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)

        # Update goal blocks information
        self.updateGoalBlocks(state)

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

        while True:
            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]

                # If no more closed doors
                if len(closedDoors) == 0:
                    # Go check if others picked up goal blocks when they said they did
                    if len(self._goal_blocks_location_followed_by_others) > 0:
                        self._phase = Phase.CHECK_PICKUP_BY_OTHER
                        return None, {}
                    # Else, search a random room that has been searched
                    else:
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
                self._navigator.reset_full()

                contents = state.get_room_objects(self._door['room_name'])
                waypoints = []

                for c in contents:
                    if 'class_inheritance' in c and 'AreaTile' in c['class_inheritance']:
                        x, y = c["location"][0], c["location"][1]
                        waypoints.append((x, y))

                self._navigator.add_waypoints(waypoints)

                # Open door
                is_open = state.get_room_doors(self._door['room_name'])[0]['is_open']

                if not is_open:
                    # Send opening door message
                    self._sendMessage(Util.openingDoorMessage(self._door['room_name']), agent_name)
                    # Go search room
                    self._phase = Phase.SEARCH_ROOM
                    # Send searching room message
                    self._sendMessage(Util.searchingThroughMessage(self._door['room_name']), agent_name)
                    return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}
                else:
                    # If another agent has openend the door in the meantime, go search room & send message
                    self._sendMessage(Util.searchingThroughMessage(self._door['room_name']), agent_name)
                    self._phase = Phase.SEARCH_ROOM

            if Phase.SEARCH_ROOM == self._phase:
                self._state_tracker.update(state)

                contents = state.get_room_objects(self._door['room_name'])

                for c in contents:
                    if "Block" in c['name']:
                        if c not in self._objects:
                            self._objects.append(c)
                            self._sendMessage(Util.foundBlockMessage(c, True), agent_name)

                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}

                # After searching room, check if there is a goal block to pick up
                self._phase = Phase.CHECK_GOAL_TO_FOLLOW

            # Check if another agent has found a goal block
            if Phase.CHECK_GOAL_TO_FOLLOW == self._phase:
                follow = None
                for loc in self._goal_blocks_locations:
                    # There is a location at which another agent found a goal block
                    # & no other agent said it's going to pick it up
                    if loc not in self._goal_blocks_locations_followed:
                        follow = loc['location']
                        self._goal_blocks_locations_followed.append(loc)
                        break

                if follow is not None:
                    self._phase = Phase.FOLLOW_PATH_TO_GOAL_BLOCK
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([follow])
                    action = self._navigator.get_move_action(self._state_tracker)
                    return action, {}

                # If there is no goal block, plan path to closed door
                else:
                    self._navigator.reset_full()
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_GOAL_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}

                # Get followed location
                location_goal = self._goal_blocks_locations_followed[-1]['location']
                # Get objects in area of location
                objs_in_area = state.get_objects_in_area(location_goal, 1, 1)
                # Get block at followed location
                l = list(filter(lambda obj: 'Block' in obj['name'] and obj['location'] == location_goal, objs_in_area))
                # If object is still there
                if len(l) > 0:
                    self._current_obj = l[0]
                    self._sendMessage(Util.pickingUpBlockSimpleMessage(self._current_obj, True), agent_name)

                    self._phase = Phase.GRAB
                    return GrabObject.__name__, {'object_id': self._current_obj['obj_id']}
                else:
                    self._phase = Phase.CHECK_GOAL_TO_FOLLOW

            if Phase.CHECK_PICKUP_BY_OTHER == self._phase:
                location_to_check = self._goal_blocks_location_followed_by_others[0]
                self._phase = Phase.GO_CHECK_PICKUP
                self._navigator.reset_full()
                self._navigator.add_waypoints([location_to_check['location']])
                action = self._navigator.get_move_action(self._state_tracker)
                return action, {}


            if Phase.GO_CHECK_PICKUP == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}

                # Get followed location
                location_to_check_followed = self._goal_blocks_location_followed_by_others[0]
                self._goal_blocks_location_followed_by_others.remove(location_to_check_followed)

                # Get objects in area of location
                objs_in_area = state.get_objects_in_area(location_to_check_followed['location'], 1, 1)

                # Get block at followed location
                l = list(filter(lambda obj: 'Block' in obj['name'] and obj['location'] == location_to_check_followed, objs_in_area))
                # If object is still there
                if len(l) > 0:
                    self._current_obj = l[0]
                    self._sendMessage(Util.pickingUpBlockSimpleMessage(self._current_obj, True), agent_name)

                    self._phase = Phase.GRAB
                    return GrabObject.__name__, {'object_id': self._current_obj['obj_id']}
                else:
                    self._phase = Phase.CHECK_GOAL_TO_FOLLOW


            if Phase.GRAB == self._phase:
                self._navigator.reset_full()

                # Blind drops object between drop zone and door of closest room
                if self._drop_location_blind is None:
                    loc = self._goal_blocks[2]['location']
                    self._drop_location_blind = (loc[0], loc[1] - 1)

                self._navigator.add_waypoints([self._drop_location_blind])

                self._phase = Phase.MOVING_BLOCK

            if Phase.MOVING_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}

                if state[agent_name]['is_carrying']:
                    self._sendMessage(Util.droppingBlockSimpleMessage(self._current_obj, self._drop_location_blind, True), agent_name)
                    return DropObject.__name__, {'object_id': self._current_obj['obj_id']}

                # After dropping block, check if there is another goal to follow
                self._phase = Phase.CHECK_GOAL_TO_FOLLOW

            if Phase.SEARCH_RANDOM_ROOM == self._phase:
                doors = [door for door in state.values()
                         if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                self._door = random.choice(doors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage(Util.moveToMessage(self._door['room_name']), agent_name)
                self._navigator.add_waypoints([doorLoc])
                # Follow path to random (opened) door, but can reuse phase
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR


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

    def foundGoalBlockUpdate(self, block, member):
        location = block['location']
        charac = block['visualization']

        # Check that agent didn't lie about size, shape and color of goal block
        for goal in self._goal_blocks:
            if goal['visualization']['size'] == charac['size'] and goal['visualization']['shape'] == charac['shape'] and \
                    goal['visualization']['colour'] == charac['colour']:

                # Save goal block locations as mentioned by other agents
                # Location + member that sent message + trust in member
                obj = {
                    "location": location,
                    "member": member,
                    "trustLevel": self._trust[member]['found']
                }
                if obj not in self._goal_blocks_locations:
                    self._goal_blocks_locations.append(obj)
                    # Sort by trust (follow first locations from most trusted team members)
                    self._goal_blocks_locations.sort(key=lambda x: x['trustLevel'], reverse=True)

    def pickUpBlockUpdate(self, block, member):
        location_goal = block['location']
        for loc in self._goal_blocks_locations:
            if loc['location'] == location_goal:
                self._goal_blocks_locations.remove(loc)

                if loc not in self._goal_blocks_location_followed_by_others:
                    self._goal_blocks_location_followed_by_others.append(loc)
                # Sort locations followed ascending on trust level
                self._goal_blocks_location_followed_by_others.sort(key=lambda x: x['trustLevel'])


    def foundBlockUpdate(self, block, member):
        return

    def dropBlockUpdate(self, block, member):
        return

    def dropGoalBlockUpdate(self, block, member):
        return


    def updateGoalBlocks(self, state):
        if len(self._goal_blocks) == 0:
            self._goal_blocks = [goal for goal in state.values()
                        if 'is_goal_block' in goal and goal['is_goal_block']]

    def _trustBlief(self, member, received, state, close_objects):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        # You can change the default value to your preference

        # Go through the seen objects
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


