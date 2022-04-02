import enum
import random
import csv
import os
import numpy as np
import json
import re
from typing import Dict
from matrx.actions.door_actions import OpenDoorAction
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message
from matrx.actions.object_actions import GrabObject, DropObject
from bw4t.BW4TBrain import BW4TBrain


class Util():
    @staticmethod
    def moveToMessage(room_name):
        return 'Moving to ' + room_name

    @staticmethod
    def openingDoorMessage(room_name):
        return 'Opening door of ' + room_name

    @staticmethod
    def searchingThroughMessage(room_name):
        return 'Searching through ' + room_name

    @staticmethod
    def foundGoalBlockMessage(data):
        item_info = dict(list(data['visualization'].items())[:3])
        return "Found goal block " + json.dumps(item_info) \
               + " at location (" + ', '.join([str(loc) for loc in data['location']]) + ")"

    @staticmethod
    def foundBlockMessage(data, is_color_blind=False):
        get_with_color = 2 if is_color_blind else 3
        item_info = dict(list(data['visualization'].items())[:get_with_color])
        return "Found block " + json.dumps(item_info) \
               + " at location (" + ', '.join([str(loc) for loc in data['location']]) + ")"

    @staticmethod
    def pickingUpBlockSimpleMessage(data, is_color_blind=False):
        get_with_color = 2 if is_color_blind else 3
        item_info = dict(list(data['visualization'].items())[:get_with_color])
        return "Picking up block " + json.dumps(item_info) \
               + " at location (" + ', '.join([str(loc) for loc in data['location']]) + ")"

    @staticmethod
    def droppingBlockSimpleMessage(data, location, is_color_blind=False):
        get_with_color = 2 if is_color_blind else 3
        item_info = dict(list(data['visualization'].items())[:get_with_color])
        return "Dropped block " + json.dumps(item_info) \
               + " at drop location (" + ', '.join([str(loc) for loc in location]) + ")"

    @staticmethod
    def pickingUpBlockMessage(data):
        item_info = dict(list(data['visualization'].items())[:3])
        return "Picking up goal block " + json.dumps(item_info) \
               + " at location (" + ', '.join([str(loc) for loc in data['location']]) + ")"

    @staticmethod
    def droppingBlockMessage(data, location):
        item_info = dict(list(data['visualization'].items())[:3])
        return "Dropped goal block " + json.dumps(item_info) \
               + " at drop location (" + ', '.join([str(loc) for loc in location]) + ")"

    @staticmethod
    def reputationMessage(trust, team_members):
        rep = {}
        for member in team_members:
            rep[member] = trust[member]['average']
        return "Reputation: " + json.dumps(rep)

    @staticmethod
    def openingDoorMessageLie(state, door):
        door_names = [room['room_name'] for room in [door for door in state.values()
                                                     if 'class_inheritance' in door and 'Door' in door[
                                                         'class_inheritance']]]
        door_names.remove(door)
        return 'Opening door of ' + random.choice(door_names)

    @staticmethod
    def moveToMessageLie(door, doors):
        room_names = [room['room_name'] for room in doors]
        room_names.remove(door)
        return 'Moving to ' + random.choice(room_names)

    @staticmethod
    def searchingThroughMessageLie(state, door):
        rooms = [room for room in state.values()
                 if 'class_inheritance' in room and 'Door' in room['class_inheritance']]
        room_names = [room['room_name'] for room in rooms]
        room_names.remove(door)
        return 'Moving to ' + random.choice(room_names)

    @staticmethod
    def foundBlockMessageLie():
        color = "%06x" % random.randint(0, 0xFFFFFF)
        message = "Found block {\"size\": 0.5, \"shape\": " + \
                  str(random.randint(0, 2)) + ", \"color\": \"#" + color + \
                  "\"} at location (" + str(random.randint(0, 12)) + ", " + str(random.randint(0, 23)) + ")"
        return message

    @staticmethod
    def pickingUpBlockMessageLie():
        color = "%06x" % random.randint(0, 0xFFFFFF)
        message = "Picking up goal block {\"size\": 0.5, \"shape\": " + \
                  str(random.randint(0, 2)) + ", \"color\": \"#" + color + \
                  "\"} at location (" + str(random.randint(0, 12)) + ", " + str(random.randint(0, 23)) + ")"
        return message

    @staticmethod
    def droppingBlockMessageLie():
        color = "%06x" % random.randint(0, 0xFFFFFF)
        message = "Dropped goal block {\"size\": 0.5, \"shape\": " + \
                  str(random.randint(0, 2)) + ", \"color\": \"#" + color + \
                  "\"} at location (" + str(random.randint(0, 12)) + ", " + str(random.randint(0, 23)) + ")"
        return message

    @staticmethod
    def update_info_general(arrayWorld, receivedMessages, teamMembers,
                            foundGoalBlockUpdate, foundBlockUpdate, pickUpBlockUpdate, pickUpBlockSimpleUpdate,
                            dropBlockUpdate, dropGoalBlockUpdate, updateRep, agent_name):
        avg_reps = {}
        for member in teamMembers:
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

                    foundGoalBlockUpdate(block, member)

                    if arrayWorld[block['location'][0], block['location'][1]] is None:
                        arrayWorld[block['location'][0], block['location'][1]] = []
                    arrayWorld[block['location'][0], block['location'][1]].append({
                        "memberName": member,
                        "block": block['visualization'],
                        "action": "found",
                    })

                elif "Found block " in msg:
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

                    foundBlockUpdate(block, member)

                    if arrayWorld[block['location'][0], block['location'][1]] is None:
                        arrayWorld[block['location'][0], block['location'][1]] = []
                    arrayWorld[block['location'][0], block['location'][1]].append({
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

                    pickUpBlockUpdate(block, member)

                    if arrayWorld[block['location'][0], block['location'][1]] is None:
                        arrayWorld[block['location'][0], block['location'][1]] = []
                    arrayWorld[block['location'][0], block['location'][1]].append({
                        "memberName": member,
                        "block": block['visualization'],
                        "action": "pick-up",
                    })

                elif "Picking up block " in msg:
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

                    pickUpBlockSimpleUpdate(block, member)

                    if arrayWorld[block['location'][0], block['location'][1]] is None:
                        arrayWorld[block['location'][0], block['location'][1]] = []
                    arrayWorld[block['location'][0], block['location'][1]].append({
                        "memberName": member,
                        "block": block['visualization'],
                        "action": "pick-up",
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
                    block['visualization'] = vis
                    block['location'] = loc

                    dropGoalBlockUpdate(block, member)

                    if arrayWorld[block['location'][0], block['location'][1]] is None:
                        arrayWorld[block['location'][0], block['location'][1]] = []
                    arrayWorld[block['location'][0], block['location'][1]].append({
                        "memberName": member,
                        "block": block['visualization'],
                        "action": "drop-off",
                    })

                elif "Dropped block " in msg:
                    pattern = re.compile("{(.* ?)}")
                    vis = re.search(pattern, msg).group(0)

                    pattern2 = re.compile("\((.* ?)\)")
                    loc = re.search(pattern2, msg).group(0)
                    loc = loc.replace("(", "[")
                    loc = loc.replace(")", "]")
                    loc = json.loads(loc)
                    vis = json.loads(vis)
                    block['visualization'] = vis
                    block['location'] = loc

                    dropBlockUpdate(block, member)

                    if arrayWorld[block['location'][0], block['location'][1]] is None:
                        arrayWorld[block['location'][0], block['location'][1]] = []
                    arrayWorld[block['location'][0], block['location'][1]].append({
                        "memberName": member,
                        "block": block['visualization'],
                        "action": "drop-off",
                    })
                elif "Reputation: " in msg:
                    pattern = re.compile("{(.* ?)}")
                    rep = re.search(pattern, msg).group(0)
                    rep = json.loads(rep)

                    for name in rep.keys():
                        if name != agent_name:
                            avg_reps[name] = rep[name]

                    updateRep(avg_reps)


class PhaseBlind(enum.Enum):
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


class ColorblindAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = PhaseBlind.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []
        self._door = None
        self._objects = []                                  # All objects found
        self._goal_blocks = []                              # Goal blocks
        self._goal_blocks_locations = []                    # Goal block locations as said by other agents
        self._goal_blocks_locations_followed = []           # Goal block locations already followed
        self._goal_blocks_locations_followed_by_others = []  # Goal block locations where other agents went
        self._trustBeliefs = []
        self._current_obj = None
        self._drop_location_blind = None
        self._trust = {}
        self._arrayWorld = None
        self.receivedMessagesIndex = 0

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        # Remove colour from any block visualization
        for obj in state:
            if "is_collectable" in state[obj] and state[obj]['is_collectable']:
                state[obj]['visualization'].pop('colour')
        return state

    def decide_on_bw4t_action(self, state: State):
        state = self.filter_bw4t_observations(state)
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)

        # Initialize trust & write beliefs
        if self._trust == {}:
            self.initialize_trust()
            self.read_trust()
        self.write_beliefs()

        # Update goal blocks information
        self.updateGoalBlocks(state)
        # Initialize arrayWorld if empty
        self._prepareArrayWorld(state)
        # Update arrayWorld with new information
        self._updateWorld(state)
        # Send reputation message
        self._sendMessage(Util.reputationMessage(self._trust, self._teamMembers), agent_name)
        # General update with info from team members
        Util.update_info_general(self._arrayWorld, receivedMessages, self._teamMembers,
                                 self.foundGoalBlockUpdate, self.foundBlockUpdate, self.pickUpBlockUpdate,
                                 self.pickUpBlockSimpleUpdate, self.dropBlockUpdate, self.dropGoalBlockUpdate,
                                 self.updateRep, self.agent_name)

        while True:
            if PhaseBlind.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]

                # If no more closed doors
                if len(closedDoors) == 0:
                    # Go check if others picked up goal blocks when they said they did
                    if len(self._goal_blocks_locations_followed_by_others) > 0:
                        self._phase = PhaseBlind.CHECK_PICKUP_BY_OTHER
                        return None, {}
                    # Else, search a random room that has been searched
                    else:
                        self._phase = PhaseBlind.SEARCH_RANDOM_ROOM
                        return None, {}

                # Randomly pick a closed door
                self._door = random.choice(closedDoors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage(Util.moveToMessage(self._door['room_name']), agent_name)

                self._navigator.add_waypoints([doorLoc])
                self._phase = PhaseBlind.FOLLOW_PATH_TO_CLOSED_DOOR

            if PhaseBlind.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = PhaseBlind.OPEN_DOOR

            if PhaseBlind.OPEN_DOOR == self._phase:
                self._navigator.reset_full()

                contents = state.get_room_objects(self._door['room_name'])
                waypoints = []

                # Add waypoint locations in room
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
                    self._phase = PhaseBlind.SEARCH_ROOM
                    # Send searching room message
                    self._sendMessage(Util.searchingThroughMessage(self._door['room_name']), agent_name)
                    return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}
                else:
                    # If another agent has opened the door in the meantime, go search room & send message
                    self._sendMessage(Util.searchingThroughMessage(self._door['room_name']), agent_name)
                    self._phase = PhaseBlind.SEARCH_ROOM

            if PhaseBlind.SEARCH_ROOM == self._phase:
                self._state_tracker.update(state)

                contents = state.get_room_objects(self._door['room_name'])

                for c in contents:
                    if "Block" in c['name']:
                        # For each new object found, remember & send "found block" message
                        if c not in self._objects:
                            self._objects.append(c)
                            self._sendMessage(Util.foundBlockMessage(c, True), agent_name)

                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}

                # After searching room, check if there is a goal block to pick up
                self._phase = PhaseBlind.CHECK_GOAL_TO_FOLLOW

            # Check if another agent has found a goal block
            if PhaseBlind.CHECK_GOAL_TO_FOLLOW == self._phase:
                follow = None
                for loc in self._goal_blocks_locations:
                    # There is a location at which another agent found a goal block
                    # & no other agent said it's going to pick it up
                    if loc['location'] not in self._goal_blocks_locations_followed:
                        follow = loc['location']
                        self._goal_blocks_locations_followed.append(follow)
                        break

                if follow is not None:
                    self._phase = PhaseBlind.FOLLOW_PATH_TO_GOAL_BLOCK
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([follow])
                    action = self._navigator.get_move_action(self._state_tracker)
                    return action, {}

                # If there is no goal block, plan path to closed door
                else:
                    self._navigator.reset_full()
                    self._phase = PhaseBlind.PLAN_PATH_TO_CLOSED_DOOR

            if PhaseBlind.FOLLOW_PATH_TO_GOAL_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}

                # Get followed location
                location_goal = self._goal_blocks_locations_followed[-1]
                # Get objects in area of location
                objs_in_area = state.get_objects_in_area(location_goal, 1, 1)
                # Get block at followed location
                l = list(filter(lambda obj: 'Block' in obj['name'] and obj['location'] == location_goal, objs_in_area))
                # If object is still there
                if len(l) > 0:
                    self._current_obj = l[0]
                    self._sendMessage(Util.pickingUpBlockSimpleMessage(self._current_obj, True), agent_name)

                    self._phase = PhaseBlind.GRAB
                    return GrabObject.__name__, {'object_id': self._current_obj['obj_id']}
                else:
                    self._phase = PhaseBlind.CHECK_GOAL_TO_FOLLOW

            if PhaseBlind.CHECK_PICKUP_BY_OTHER == self._phase:
                location_to_check = self._goal_blocks_locations_followed_by_others[0]
                self._phase = PhaseBlind.GO_CHECK_PICKUP
                self._navigator.reset_full()
                self._navigator.add_waypoints([location_to_check['location']])
                action = self._navigator.get_move_action(self._state_tracker)
                return action, {}

            if PhaseBlind.GO_CHECK_PICKUP == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}

                # Get followed location & remove from array
                location_to_check_followed = self._goal_blocks_locations_followed_by_others[0]
                self._goal_blocks_locations_followed_by_others.remove(location_to_check_followed)

                # Get objects in area of location
                objs_in_area = state.get_objects_in_area(location_to_check_followed['location'], 1, 1)

                # Get block at followed location
                l = list(filter(lambda obj: 'Block' in obj['name'] and obj['location'] == location_to_check_followed,
                                objs_in_area))
                # If object is still there
                if len(l) > 0:
                    self._current_obj = l[0]
                    self._sendMessage(Util.pickingUpBlockSimpleMessage(self._current_obj, True), agent_name)

                    self._phase = PhaseBlind.GRAB
                    return GrabObject.__name__, {'object_id': self._current_obj['obj_id']}
                else:
                    self._phase = PhaseBlind.CHECK_GOAL_TO_FOLLOW

            if PhaseBlind.GRAB == self._phase:
                self._navigator.reset_full()

                # Custom drop location = above drop zone
                if self._drop_location_blind is None:
                    loc = self._goal_blocks[2]['location']
                    self._drop_location_blind = (loc[0], loc[1] - 1)

                self._navigator.add_waypoints([self._drop_location_blind])
                self._phase = PhaseBlind.MOVING_BLOCK

            if PhaseBlind.MOVING_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}

                if state[agent_name]['is_carrying']:
                    self._sendMessage(Util.droppingBlockSimpleMessage(self._current_obj, self._drop_location_blind, True), agent_name)
                    return DropObject.__name__, {'object_id': self._current_obj['obj_id']}

                # After dropping block, check if there is another goal to follow
                self._phase = PhaseBlind.CHECK_GOAL_TO_FOLLOW

            if PhaseBlind.SEARCH_RANDOM_ROOM == self._phase:
                doors = [door for door in state.values()
                         if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                self._door = random.choice(doors)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage(Util.moveToMessage(self._door['room_name']), agent_name)
                self._navigator.add_waypoints([doorLoc])
                # Follow path to random (opened or closed) door, but can reuse phase
                self._phase = PhaseBlind.FOLLOW_PATH_TO_CLOSED_DOOR

    ####### MESSAGES #######
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

    ###### UPDATE METHODS ########
    def foundGoalBlockUpdate(self, block, member):
        # Update trust for objects already in goal_blocks_locations
        for obj in self._goal_blocks_locations:
            member = obj['member']
            trust = self._trust[member]['found'] if self._trust[member]['verified'] > 2 else self._trust[member]['rep']
            obj['trustLevel'] = trust

        location = block['location']
        charac = block['visualization']

        # Check that agent didn't lie about size, shape and color of goal block
        for goal in self._goal_blocks:
            if goal['visualization']['size'] == charac['size'] and goal['visualization']['shape'] == charac['shape'] and \
                    goal['visualization']['colour'] == charac['colour']:

                # Save goal block locations as mentioned by other agents
                # Location + member that sent message + trust in member for finding blocks action
                obj = {
                    "location": location,
                    "member": member,
                    "trustLevel": self._trust[member]['found'] if self._trust[member]['verified'] > 2 else self._trust[member]['rep']
                }
                if obj not in self._goal_blocks_locations:
                    self._goal_blocks_locations.append(obj)

        # Sort by trust (follow first locations from most trusted team members)
        self._goal_blocks_locations.sort(key=lambda x: x['trustLevel'], reverse=True)

    def pickUpBlockUpdate(self, block, member):
        self.removeLocationsFollowedByOther(block)

    def pickUpBlockSimpleUpdate(self, block, member):
        self.removeLocationsFollowedByOther(block)

    def removeLocationsFollowedByOther(self, block):
        # Remove locations followed by others from self.goal_blocks_locations
        # & add to self.goal_blocks_locations_followed_by_others
        location_goal = block['location']
        for loc in self._goal_blocks_locations:
            if loc['location'] == location_goal:
                self._goal_blocks_locations.remove(loc)
                if loc not in self._goal_blocks_locations_followed_by_others:
                    self._goal_blocks_locations_followed_by_others.append(loc)

        # Update trust values for all objects
        for obj in self._goal_blocks_locations_followed_by_others:
            member = obj['member']
            trust = self._trust[member]['pick-up'] if self._trust[member]['verified'] > 2 else self._trust[member]['rep']
            obj['trustLevel'] = trust

        # Sort locations followed ascending on trust level
        # These locations are going to be checked, so go ascending on trust level (first check least trusted agent)
        self._goal_blocks_locations_followed_by_others.sort(key=lambda x: x['trustLevel'])

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

    #### ArrayWorld #####
    def _updateWorld(self, state):
        agent_location = state[self.agent_id]['location']
        closeObjects = state.get_objects_in_area((agent_location[0] - 1, agent_location[1] - 1),
                                                 bottom_right=(agent_location[0] + 1, agent_location[1] + 1))
        # Filter out only blocks
        close_blocks = None
        if closeObjects is not None:
            close_blocks = [obj for obj in closeObjects
                           if 'CollectableBlock' in obj['class_inheritance']]

        # Update trust beliefs for team members
        self._trustBelief(state, close_blocks)

        # Add average trust value
        for member in self._teamMembers:
            avg = 0
            for key in self._trust[member].keys():
                if key in ["pick-up", "drop-off", "found"]:
                    avg += self._trust[member][key] / 3.0
            self._trust[member]['average'] = avg

    def _prepareArrayWorld(self, state):
        # Get world shape
        world_shape = state['World']['grid_shape']
        # Initialize empty arrayWorld same shape as world
        if self._arrayWorld is None:
            self._arrayWorld = np.empty(world_shape, dtype=list)
            for x in range(world_shape[0]):
                for y in range(world_shape[1]):
                    self._arrayWorld[x, y] = []

    ######### TRUST #############
    def read_trust(self):
        # agentname_trust.csv
        file_name = self.agent_id + '_trust.csv'
        if os.path.exists(file_name):
            with open(file_name, newline='') as file:
                reader = csv.reader(file, delimiter=',')
                for row in reader:
                    if row:
                        self._trust[row[0]] = {"pick-up": float(row[1]),
                                               "drop-off": float(row[2]),
                                               "found": float(row[3]),
                                               "average": float(row[4]),
                                               "rep": float(row[5]),
                                               "verified": float(row[6])}
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
            names = self._trust.keys()
            for name in names:
                row = self._trust[name]
                row['name'] = name
                writer.writerow(row)

    def _trustBelief(self, state, close_objects):
        agentLocation = state[self.agent_id]['location']
        (x, y) = agentLocation
        messages = self._arrayWorld[x][y]
        self._arrayWorld[x][y] = []
        if len(messages) > 0:  # there is some sort of block interaction!
            realBlock = self.getObjectAtLocation(close_objects, (x, y))
            if realBlock == "MultipleObj":
                return
            if realBlock is None:  # no actual block there so interaction must end with pickup to be valid!
                self.checkPickUpInteraction(messages)
            else:  # block is there so interaction must end with found or drop-off to be valid!
                self.checkFoundInteraction(messages, realBlock)

    # Get object(s) at a given location
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

    def checkPickUpInteraction(self, interactions):
        # Assume interactions are for the same type of block(same visualization)
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

    def increaseDecreaseTrust(self, members, isIncrease, block=None):
        val = -0.1
        if isIncrease:
            val = 0.1
        for member in members:
            if block is not None:
                val = self.check_same_visualizations(block['visualization'], member[2])
            self._trust[member[0]][member[1]] = min(max(round(self._trust[member[0]][member[1]] + val, 3), 0), 1)
            self._trust[member[0]]['verified'] += 1

    # Check if 2 block visualizations have the same values for shape and colour
    def check_same_visualizations(self, vis1, vis2):
        shape = 0
        colour = 0
        if "shape" in vis1 and "shape" in vis2:
            shape = 0.05 if vis1['shape'] == vis2['shape'] else -0.05
        if "colour" in vis1 and "colour" in vis2:
            colour = 0.05 if vis1['colour'] == vis2['colour'] else -0.05
        return shape + colour

    # Update reputation values
    def updateRep(self, avg_reps):
        nr_team_mates = len(self._teamMembers)
        for member in avg_reps.keys():
            self._trust[member]['rep'] = (avg_reps[member] + (
                        self._trust[member]['rep'] * (nr_team_mates - 1))) / nr_team_mates


class PhaseLiar(enum.Enum):
    FOLLOW_PATH_TO_DOOR = 1,
    OPEN_DOOR = 2,
    PLAN_PATH_TO_UNSEARCHED_DOOR = 3,
    SEARCH_ROOM = 4,
    FIND_BLOCK = 5,
    GRAB = 6,
    MOVE_TO_OBJECT = 7,
    MOVING_BLOCK = 8,
    CHECK_GOAL_ZONE = 9,
    CHOOSE_NEXT_MOVE = 10

class LiarAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = PhaseLiar.PLAN_PATH_TO_UNSEARCHED_DOOR
        self._teamMembers = []
        self._goal_objects_found = []
        self._goal_objects = None
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

        while True:
            if PhaseLiar.PLAN_PATH_TO_UNSEARCHED_DOOR == self._phase:
                self._navigator.reset_full()
                # check each room in the given order
                doors = [door for door in state.values()
                         if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

                if len(self._searched_doors_index) <= 0:
                    check_next_to_goal_zone = random.uniform(0, 1)
                    if check_next_to_goal_zone <= 0.5:
                        self._phase = PhaseLiar.CHECK_GOAL_ZONE
                        return None, {}
                    else:
                        self._door = random.choice(doors)
                else:
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
                else:
                    self._sendMessage(Util.moveToMessageLie(self._door['room_name'], doors), agent_name)
                self._navigator.add_waypoints([door_location])
                self._phase = PhaseLiar.FOLLOW_PATH_TO_DOOR

            # Follow path to door
            if PhaseLiar.FOLLOW_PATH_TO_DOOR == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}

                # Check if the door is open
                if not self._door['is_open']:
                    self._phase = PhaseLiar.OPEN_DOOR
                else:
                    self._phase = PhaseLiar.SEARCH_ROOM

            if PhaseLiar.OPEN_DOOR == self._phase:
                # Send message with a probability of 0.8 to lie
                lie = random.uniform(0, 1)
                if lie <= 0.2:
                    self._sendMessage(Util.openingDoorMessage(self._door['room_name']), agent_name)
                else:
                    self._sendMessage(Util.openingDoorMessageLie(state, self._door['room_name']), agent_name)

                self._phase = PhaseLiar.SEARCH_ROOM
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if PhaseLiar.SEARCH_ROOM == self._phase:
                self._navigator.reset_full()
                room_area = []
                for area in state.get_room_objects(self._door['room_name']):
                    if "wall" not in area['name'] and not self._teamMembers.__contains__(area['name']):
                        room_area.append((area["location"][0], area["location"][1]))

                # Send message with a probability of 0.8 to lie
                lie = random.uniform(0, 1)
                if lie <= 0.2:
                    self._sendMessage(Util.searchingThroughMessage(self._door['room_name']), agent_name)
                else:
                    self._sendMessage(Util.searchingThroughMessageLie(state, self._door['room_name']), agent_name)

                self._navigator.add_waypoints(room_area)
                self._phase = PhaseLiar.FIND_BLOCK

            if PhaseLiar.FIND_BLOCK == self._phase:
                self._state_tracker.update(state)

                # Check if the objects from the room are the searched blocks
                contents = state.get_room_objects(self._door['room_name'])
                for c in contents:
                    for i in range(len(self._goal_objects)):
                        if c['visualization']['colour'] == self._goal_objects[i]['visualization']['colour'] and \
                                c['visualization']['shape'] == self._goal_objects[i]['visualization']['shape'] and \
                                c['visualization']['size'] == self._goal_objects[i]['visualization']['size'] and \
                                not c['is_goal_block'] and not c['is_drop_zone']:
                            if i == 0:
                                if not self._objects.__contains__(c):
                                    # Send message with a probability of 0.8 to lie
                                    lie = random.uniform(0, 1)
                                    if lie <= 0.2:
                                        self._sendMessage(Util.foundGoalBlockMessage(c), agent_name)
                                    else:
                                        self._sendMessage(Util.foundBlockMessageLie(), agent_name)
                                    self._objects.append(c)
                                # If a goal block has been found move in top of it
                                self._phase = PhaseLiar.MOVE_TO_OBJECT
                                self._current_obj = c
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([c['location']])
                                action = self._navigator.get_move_action(self._state_tracker)
                                return action, {}
                            else:
                                self._goal_objects_found.append(c)

                    if "Block" in c['name']:
                        if not self._objects.__contains__(c):
                            # Send message with a probability of 0.8 to lie
                            lie = random.uniform(0, 1)
                            if lie <= 0.2:
                                self._sendMessage(Util.foundGoalBlockMessage(c), agent_name)
                            else:
                                self._sendMessage(Util.foundBlockMessageLie(), agent_name)
                            self._objects.append(c)

                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                # If no object have been found search another room
                self._phase = PhaseLiar.PLAN_PATH_TO_UNSEARCHED_DOOR

            if PhaseLiar.MOVE_TO_OBJECT == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = PhaseLiar.GRAB
                # Grab the object the the agent is standing on top of
                return GrabObject.__name__, {'object_id': self._current_obj['obj_id']}

            if PhaseLiar.GRAB == self._phase:
                # Check if the agent was abel to grab the object
                if not state[agent_name]['is_carrying']:
                    self._phase = PhaseLiar.PLAN_PATH_TO_UNSEARCHED_DOOR
                    return None, {}
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goal_objects[0]['location']])
                self._phase = PhaseLiar.MOVING_BLOCK

                # Send message with a probability of 0.8 to lie
                lie = random.uniform(0, 1)
                if lie <= 0.2:
                    self._sendMessage(Util.pickingUpBlockMessage(self._current_obj), agent_name)
                else:
                    self._sendMessage(Util.pickingUpBlockMessageLie(), agent_name)

            if PhaseLiar.MOVING_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
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
                    self._phase = PhaseLiar.CHOOSE_NEXT_MOVE
                    return DropObject.__name__, {'object_id': self._current_obj['obj_id']}
                else:
                    self._phase = PhaseLiar.CHOOSE_NEXT_MOVE

            if PhaseLiar.CHOOSE_NEXT_MOVE == self._phase:
                self._phase = PhaseLiar.PLAN_PATH_TO_UNSEARCHED_DOOR
                # if there is an goal object that we have previously found
                if len(self._goal_objects_found) > 0 and len(self._goal_objects) > 0:
                    for obj in self._goal_objects_found:
                        if obj['visualization']['colour'] == self._goal_objects[0]['visualization']['colour'] and \
                                obj['visualization']['shape'] == self._goal_objects[0]['visualization']['shape']:
                            # Use this method to check whether any agent had moved the block, and act accordingly
                            (picked, location) = \
                                self.checkLocationOfBlock(receivedMessages, obj['location'],
                                                          obj['visualization'])
                            if picked is False:
                                location = [obj['location']]
                            elif picked is True and location is None:

                                self._phase = PhaseLiar.PLAN_PATH_TO_UNSEARCHED_DOOR
                                break

                            self._navigator.reset_full()
                            self._goal_objects_found.remove(obj)
                            if isinstance(location, tuple):
                                self._navigator.add_waypoints([location])
                            else:
                                self._navigator.add_waypoints(location)
                            self._phase = PhaseLiar.MOVE_TO_OBJECT
                            self._current_obj = obj

            if PhaseLiar.CHECK_GOAL_ZONE == self._phase:
                # Check if any block has been misplaced in the goal zone area
                loc = [goal for goal in state.values()
                       if 'is_goal_block' in goal and goal['is_goal_block']][2]['location']
                goal_zone = (loc[0], loc[1] - 1)
                self._navigator.add_waypoints([goal_zone])
                self._state_tracker.update(state)
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
                                self._phase = PhaseLiar.MOVE_TO_OBJECT
                                self._current_obj = c
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([c['location']])
                                action = self._navigator.get_move_action(self._state_tracker)
                                return action, {}
                            else:
                                self._goal_objects_found.append(c)

                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = PhaseLiar.PLAN_PATH_TO_UNSEARCHED_DOOR

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
                if str(message).startswith("Picking up ") and str(message).endswith(str(knownLocation)) and\
                        self._trust[member]['average'] >= 0.7:
                    shape = "\"shape\": " + str(visualzation['shape'])
                    for m in reversed(receivedMessages[member]):
                        if str(m).startswith("Dropped ") and str(m).__contains__(shape):
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
        if self._goal_objects is None:
            self._goal_objects = [goal for goal in state.values()
                                  if 'is_goal_block' in goal and goal['is_goal_block']]

    def pickUpBlockSimpleUpdate(self, block, member):
        return

    def updateRep(self, avg_reps):
        nr_team_mates = len(self._teamMembers)
        for member in avg_reps.keys():
            self._trust[member]['rep'] = (avg_reps[member] + (
                        self._trust[member]['rep'] * (nr_team_mates - 1))) / nr_team_mates


class PhaseLazy(enum.Enum):
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
    DROP_OBJECT_NEAR_GOAL = 13
    RESET = 14
    PLAN_PATH_TO_DROP_ZONE = 15
    GO_TO_DROP_ZONE = 16

class LazyAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = PhaseLazy.START
        self._teamMembers = []

    # noinspection PyFinal
    def initialize(self):
        super().initialize()
        self._door = None
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

        self.blocks = {}
        self.current = 1

        self._trust = {}

        self._can_be_lazy = True

        self._arrayWorld = None

        self.moving_to = None

        self.receivedMessagesIndex = 0

        self.was_lazy = False

        self.foundBlocks = []

        # might not be needed at all
        # self._objects = set()

    def filter_bw4t_observations(self, state):
        return state

    # noinspection PyTypeChecker
    def decide_on_bw4t_action(self, state: State):
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)

        # -----------------TRUST-----------------
        if self._trust == {}:
            self.initialize_trust()
            self.read_trust()
        self.write_beliefs()
        # ------------------------------------

        self._prepareArrayWorld(state)

        self._updateWorld(state)

        self._sendMessage(Util.reputationMessage(self._trust, self._teamMembers), agent_name)

        Util.update_info_general(self._arrayWorld, receivedMessages, self._teamMembers,
                                 self.foundGoalBlockUpdate, self.foundBlockUpdate, self.pickUpBlockUpdate,
                                 self.pickUpBlockSimpleUpdate, self.dropBlockUpdate, self.dropGoalBlockUpdate,
                                 self.updateRep, self.agent_name)

        while True:
            if PhaseLazy.START == self._phase:

                # Initialize block dictionary
                self.blocks["1"] = {}
                self.blocks["2"] = {}
                self.blocks["3"] = {}

                self.blocks["1"]["visualization"] = state['Collect_Block']['visualization']
                self.blocks["2"]["visualization"] = state['Collect_Block_1']['visualization']
                self.blocks["3"]["visualization"] = state['Collect_Block_2']['visualization']

                self.blocks["1"]["idx"] = 1
                self.blocks["2"]["idx"] = 2
                self.blocks["3"]["idx"] = 3

                self.blocks["1"]["drop"] = state['Collect_Block']['location']
                self.blocks["2"]["drop"] = state['Collect_Block_1']['location']
                self.blocks["3"]["drop"] = state['Collect_Block_2']['location']

                self.blocks["1"]["locs"] = []
                self.blocks["2"]["locs"] = []
                self.blocks["3"]["locs"] = []

                self._phase = PhaseLazy.PLAN_PATH_TO_CLOSED_DOOR

            if PhaseLazy.PLAN_PATH_TO_CLOSED_DOOR == self._phase:

                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                        'is_open']]

                # Randomly pick a closed door
                if len(closedDoors) != 0:
                    self._door = random.choice(closedDoors)
                # otherwise pick a random door
                else:
                    self._door = random.choice([door for door in state.values()
                                                if 'class_inheritance' in door and 'Door' in door['class_inheritance']])
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage(Util.moveToMessage(self._door['room_name']), agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = PhaseLazy.FOLLOW_PATH_TO_CLOSED_DOOR

            if PhaseLazy.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}

                self._phase = PhaseLazy.OPEN_DOOR

            if PhaseLazy.OPEN_DOOR == self._phase:

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
                self._phase = PhaseLazy.SEARCH_ROOM

            if PhaseLazy.SEARCH_ROOM == self._phase:
                # -------------------LAZYNESS----------------
                if self.__is_lazy() and self._can_be_lazy:
                    self._phase = random.choice(
                        [PhaseLazy.PLAN_PATH_TO_CLOSED_DOOR, PhaseLazy.PLAN_PATH_TO_OBJECT, PhaseLazy.PLAN_PATH_TO_DROP_ZONE])
                    self._can_be_lazy = False
                # -------------------------------------------

                self._state_tracker.update(state)
                contents = state.get_room_objects(self._door['room_name'])
                contents = list(filter(lambda obj: 'Block' in obj['name'], contents))

                # check objects in room
                for c in contents:
                    for block in self.blocks.values():
                        # check if is goal block
                        if self.check_same_visualizations(c['visualization'], block['visualization']) and not c[
                            'is_goal_block'] and not c['is_drop_zone'] and not self.already_delivered(c):

                            if c not in self.foundBlocks:
                                self.foundBlocks.append(c)

                            if block["idx"] == self.current:
                                self._phase = PhaseLazy.PLAN_PATH_TO_OBJECT
                                # found an object, finished searching task, now it can be lazy again
                                block['locs'].append(c['location'])
                                self._can_be_lazy = True
                                self.sendFoundBlockMessages()
                                return None, {}
                            else:
                                block['locs'].append(c['location'])

                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}
                self.sendFoundBlockMessages()
                self._phase = PhaseLazy.PLAN_PATH_TO_CLOSED_DOOR
                # finished searching room, it can now be lazy again
                self._can_be_lazy = True

            if PhaseLazy.PLAN_PATH_TO_OBJECT == self._phase:
                # -------------------LAZYNESS----------------
                if self.__is_lazy() and self._can_be_lazy:
                    self._phase = random.choice([PhaseLazy.PLAN_PATH_TO_CLOSED_DOOR, PhaseLazy.PLAN_PATH_TO_DROP_ZONE])
                    self._can_be_lazy = False
                # -------------------------------------------
                # check if it has a location
                if len(self.blocks[str(self.current)]['locs']) != 0:
                    self._navigator.reset_full()
                    choice = random.choice(self.blocks[str(self.current)]['locs'])
                    self._navigator.add_waypoints([choice])
                    self.moving_to = choice
                    self._phase = PhaseLazy.MOVE_TO_OBJECT
                else:
                    self._phase = PhaseLazy.PLAN_PATH_TO_CLOSED_DOOR

            if PhaseLazy.MOVE_TO_OBJECT == self._phase:

                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = PhaseLazy.GRAB
                self._can_be_lazy = True

            if PhaseLazy.GRAB == self._phase:

                location_goal = self.moving_to
                # Get objects in area of location
                objs_in_area = state.get_objects_in_area(location_goal, 1, 1)
                # Get block at followed location
                l = list(filter(lambda obj: 'Block' in obj['name'] and obj[
                    'location'] == location_goal and self.check_same_visualizations(obj['visualization'],
                                                                                    self.blocks[str(self.current)][
                                                                                        'visualization']),
                                objs_in_area))

                if len(l) != 0:
                    self._sendMessage(Util.pickingUpBlockMessage(l[0]), agent_name)
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self.blocks[str(self.current)]['drop']])
                    self._phase = PhaseLazy.MOVING_BLOCK

                    # remove possible location from dict
                    self.blocks[str(self.current)]['locs'].remove(self.moving_to)
                    self.moving_to = None

                    return GrabObject.__name__, {'object_id': l[0]['obj_id']}

                self._phase = PhaseLazy.PLAN_PATH_TO_CLOSED_DOOR

            if PhaseLazy.MOVING_BLOCK == self._phase:

                # -------------------LAZYNESS----------------
                if self.__is_lazy() and self._can_be_lazy and not self.was_lazy:
                    self._phase = random.choice([PhaseLazy.PLAN_PATH_TO_CLOSED_DOOR, PhaseLazy.PLAN_PATH_TO_DROP_ZONE])
                    self._can_be_lazy = False
                    self.was_lazy = True

                    if state[agent_name]['is_carrying']:
                        # remember where it was dropped
                        self.blocks[str(self.current)]['locs'].append(state[agent_name]['location'])
                        self._sendMessage(
                            Util.droppingBlockMessage(self.blocks[str(self.current)], state[agent_name]['location']),
                            agent_name)

                        return DropObject.__name__, {'object_id': state[agent_name]['is_carrying'][0]['obj_id']}
                # -------------------------------------------
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}

                self._phase = PhaseLazy.DROP_OBJECT

            if PhaseLazy.DROP_OBJECT == self._phase:
                location_goal = self.blocks["2"]['drop']
                # Get objects in area of location
                objs_in_area = state.get_objects_in_area(location_goal, 1, 1)

                #  not working
                for goal in self.blocks.values():
                    for obj in objs_in_area:
                        if self.check_same_visualizations(obj, goal) and obj['location'] == goal['drop']:
                            if self.current < goal['idx']:
                                self.PhaseLazy = PhaseLazy.DROP_OBJECT_NEAR_GOAL
                                self._navigator.reset_full()
                                location_goal = self.blocks["3"]['drop']
                                location_goal[1] += 1
                                self._navigator.add_waypoints([location_goal])
                                return None, {}

                if state[agent_name]['is_carrying']:
                    self._sendMessage(
                        Util.droppingBlockMessage(self.blocks[str(self.current)], state[agent_name]['location']),
                        agent_name)
                    self._can_be_lazy = True
                    # increase current index
                    if self.current < 3:
                        self.current += 1
                    self._phase = PhaseLazy.RESET
                    self.was_lazy = False
                    return DropObject.__name__, {'object_id': state[agent_name]['is_carrying'][0]['obj_id']}
                else:
                    if self.current < 3:
                        self.current += 1
                    else:
                        self._phase = PhaseLazy.STOP

                    self._phase = PhaseLazy.RESET
                    return None, {}

            # not in use
            if PhaseLazy.DROP_OBJECT_NEAR_GOAL == self._phase:
                print("drop obj near goal?")
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}

                self._phase = PhaseLazy.RESET
                self._can_be_lazy = True
                self._sendMessage(
                    Util.droppingBlockMessage(self.blocks[str(self.current)], state[agent_name]['location']),
                    agent_name)
                return DropObject.__name__, {'object_id': state[agent_name]['is_carrying'][0]['obj_id']}

            if PhaseLazy.RESET == self._phase:
                # either go to object or to a closed door
                if len(self.blocks[str(self.current)]['locs']) != 0:
                    self._phase = PhaseLazy.PLAN_PATH_TO_OBJECT

                else:
                    self._phase = PhaseLazy.PLAN_PATH_TO_CLOSED_DOOR

            # check drop-zone for blocks dropped by blind
            if PhaseLazy.PLAN_PATH_TO_DROP_ZONE == self._phase:
                self._navigator.reset_full()
                loc = self.blocks["3"]['drop']
                as_list = list(loc)
                as_list[1] -= 1
                loc = tuple(as_list)
                self.moving_to = loc
                self._navigator.add_waypoints([loc])
                self._phase = PhaseLazy.GO_TO_DROP_ZONE

            if PhaseLazy.GO_TO_DROP_ZONE == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}

                location_goal = self.moving_to
                # Get objects in area of location
                objs_in_area = state.get_objects_in_area(location_goal, 1, 1)
                # Get block at followed location
                l = list(filter(lambda obj: 'Block' in obj['name'] and obj[
                    'location'] == location_goal and self.check_same_visualizations(obj['visualization'],
                                                                                    self.blocks[str(self.current)][
                                                                                        'visualization']),
                                objs_in_area))

                if len(l) != 0:
                    self._sendMessage(Util.pickingUpBlockMessage(l[0]), agent_name)
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self.blocks[str(self.current)]['drop']])
                    self._phase = PhaseLazy.MOVING_BLOCK

                    # remove possible location froim dict
                    if self.moving_to in self.blocks[str(self.current)]['locs']:
                        self.blocks[str(self.current)]['locs'].remove(self.moving_to)
                    self.moving_to = None

                    return GrabObject.__name__, {'object_id': l[0]['obj_id']}

                self.moving_to = None
                self._phase = PhaseLazy.RESET

            if PhaseLazy.STOP == self._phase:
                return None, {}

    def __is_lazy(self):
        return random.randint(0, 1) == 1

    def already_delivered(self, o1):
        for block in self.blocks.values():
            if o1['visualization'] == block['visualization']:
                return True
        return False

    def check_same_visualizations(self, vis1, vis2):
        shape = False
        colour = False

        if "shape" in vis1 and "shape" in vis2:
            shape = True if vis1['shape'] == vis2['shape'] else False

        if "colour" in vis1 and "colour" in vis2:
            colour = True if vis1['colour'] == vis2['colour'] else False

        return shape and colour

    def check_same_visualization(self, vis1, vis2):
        shape = 0
        colour = 0

        if "shape" in vis1 and "shape" in vis2:
            shape = 0.05 if vis1['shape'] == vis2['shape'] else -0.05

        if "colour" in vis1 and "colour" in vis2:
            colour = 0.05 if vis1['colour'] == vis2['colour'] else -0.05

        return shape + colour

    ####################################################
    #                       MESSAGES
    ####################################################

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def dropGoalBlockUpdate(self, block, member):
        if (self._trust[member]['pick-up'] > 0.7 and self._trust[member]['verified'] > 2) or self._trust[member][
            'rep'] > 0.7:
            # check if goal block and dropped at drop-off zone
            current_block = self.blocks[str(self.current)]

            if self.check_same_visualizations(current_block['visualization'], block['visualization']) and \
                    current_block['drop'][0] == block['location'][0] and current_block['drop'][1] == block['location'][
                1]:
                self._phase = PhaseLazy.DROP_OBJECT
            else:
                i = self.current
                while i < 4:
                    current_block = self.blocks[str(i)]
                    # if dropped somewhere else add to locations
                    if self.check_same_visualizations(['visualization'], block['visualization']):
                        self.blocks[str(current_block["idx"])]['locs'].append(block['location'])
                    i += 1

    def pickUpBlockUpdate(self, block, member):
        if (self._trust[member]['pick-up'] > 0.7 and self._trust[member]['verified'] > 2) or self._trust[member][
            'rep'] > 0.7:
            for goal in self.blocks.values():
                for loc in goal['locs']:
                    # if goal in list of loctions, remove
                    if self.check_same_visualizations(goal['visualization'], block['visualization']) and loc[0] == \
                            block['location'][0] and loc[1] == block['location'][1]:
                        self.blocks[str(goal["idx"])]['locs'].remove(loc)

    def foundGoalBlockUpdate(self, block, member):
        if (self._trust[member]['pick-up'] > 0.7 and self._trust[member]['verified'] > 2) or self._trust[member][
            'rep'] > 0.7:
            for goal in self.blocks.values():
                if self.check_same_visualizations(goal['visualization'], block['visualization']):
                    self.blocks[str(goal["idx"])]['locs'].append((block['location'][0], block['location'][1]))

    def foundBlockUpdate(self, block, member):
        if (self._trust[member]['pick-up'] > 0.7 and self._trust[member]['verified'] > 2) or self._trust[member][
            'rep'] > 0.7:
            # self._objects.add(block)
            return

    def dropBlockUpdate(self, block, member):
        return

    def pickUpBlockSimpleUpdate(self, block, member):
        return

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

    def sendFoundBlockMessages(self):
        for block in self.foundBlocks:
            self._sendMessage(Util.foundGoalBlockMessage(block), self.agent_name)
        self.foundBlocks = []

    ####################################################
    #                       TRUST
    ####################################################

    def _trustBlief(self, state, close_objects):
        agentLocation = state[self.agent_id]['location']
        (x, y) = agentLocation
        messages = self._arrayWorld[x][y]
        self._arrayWorld[x][y] = []
        if len(messages) > 0:  # there is some sort of block interaction!
            realBlock = self.getObjectAtLocation(close_objects, (x, y))
            if realBlock == "MultipleObj":
                return
            if realBlock is None:  # no actual block there so interaction must end with pickup to be valid!
                self.checkPickUpInteraction(messages)
            else:  # block is there so interaction must end with found or drop-off to be valid!
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
                val = self.check_same_visualization(block['visualization'], member[2])
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
        self._trustBlief(state, closeBlocks)

        # add average trust
        for member in self._teamMembers:
            avg = 0
            for key in self._trust[member].keys():
                if key in ["pick-up", "drop-off", "found"]:
                    avg += self._trust[member][key] / 3.0
            self._trust[member]['average'] = avg

    def _prepareArrayWorld(self, state):
        worldShape = state['World']['grid_shape']
        if self._arrayWorld is None:
            self._arrayWorld = np.empty(worldShape, dtype=list)
            for x in range(worldShape[0]):
                for y in range(worldShape[1]):
                    self._arrayWorld[x, y] = []

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

    def updateRep(self, avg_reps):
        nr_team_mates = len(self._teamMembers)
        for member in avg_reps.keys():
            self._trust[member]['rep'] = (avg_reps[member] + (
                        self._trust[member]['rep'] * (nr_team_mates - 1))) / nr_team_mates

class PhaseStrong(enum.Enum):
    DROP_BLOCK = 9,
    FOLLOW_PATH_TO_GOAL = 8,
    PICKUP_BLOCK = 7,
    FOUND_BLOCK = 6,
    SEARCH_BLOCK = 5,
    PREPARE_ROOM = 4,
    PLAN_PATH_TO_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3,
    CHECK_ABOVE_GOAL_DROP = 10,
    PATH_ABOVE_GOAL_DROP = 11


class StrongAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._check_above_goal = (False, False)
        self._currentIndex = 0
        self._foundGoalBlocks = None
        self._currentRoomObjects = None
        self._door = None
        self._phase = PhaseStrong.PLAN_PATH_TO_DOOR
        self._teamMembers = []
        self._goalBlocks = []
        self._trust = {}
        self._arrayWorld = None
        self._holdingBlocks = []
        self._doors = None
        self._doorIndex = 0
        self._blockToPick = None
        self.receivedMessagesIndex = 0
        self._drop_location_above_goal = None
        self._ticks = 0
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
        # -----------------TRUST-----------------
        if self._trust == {}:
            self.initialize_trust()
            self.read_trust()
        self.write_beliefs()

        # ------------------------------------
        # initialise the game map in an array for trust checking
        self._prepareArrayWorld(state)
        #intialise the goal blocks and the found blocks array
        self.updateGoalBlocks(state)
        #always update the holding blocks each turn
        self._holdingBlocks = state.get_self()['is_carrying']
        #Prepare the order of rooms to seach in
        self._prepareDoors(state)
        #Update the world array based on recieved messages, also change the trust belief
        self._updateWorld(state)
        #process messages from other players
        self._sendMessage(Util.reputationMessage(self._trust, self._teamMembers), agent_name)
        Util.update_info_general(self._arrayWorld, receivedMessages, self._teamMembers,
                                 self.foundGoalBlockUpdate, self.foundBlockUpdate,
                                 self.pickUpBlockUpdate, self.pickUpBlockSimpleUpdate, self.dropBlockUpdate, self.dropGoalBlockUpdate, self.updateRep, agent_name)
        #set the flags for checking above the goal
        self.setDataAboveGoal()
        # keep track of ticks
        self._ticks += 1
        while True:
            #If a block I hold is already dropped at the goal I will drop it
            droppableBlock = self.dropOldGoalBlock()
            if droppableBlock is not None:
                self._sendMessage(Util.droppingBlockMessage(
                    droppableBlock, self._goalBlocks[self._currentIndex - 1]['location']), agent_name)
                return DropObject.__name__, {
                    'object_id': droppableBlock['obj_id']}
            ##########################################
            #Check if I can find the next block to drop directly
            self.checkNextDropPossibility()
            #Check if a block I hold can be dropped at the goal
            if self.checkCurrentBlockDrop():
                self._phase = PhaseStrong.FOLLOW_PATH_TO_GOAL
            #Choose which room to go to. If all rooms have been visited choose a random one
            if PhaseStrong.PLAN_PATH_TO_DOOR == self._phase:
                self.PLAN_PATH_TO_DOOR_logic(agent_name)
            #Go to the room
            if PhaseStrong.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = PhaseStrong.OPEN_DOOR

            if PhaseStrong.OPEN_DOOR == self._phase:
                self._navigator.reset_full()
                self._phase = PhaseStrong.PREPARE_ROOM
                # Open door
                self._sendMessage(Util.openingDoorMessage(self._door['room_name']), agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}
            #Choose the area tiles in the room so you can walk in the whole room area
            if PhaseStrong.PREPARE_ROOM == self._phase:
                self.PREPARE_ROOM_logic(state, agent_name)
            #Walk on each area tile in the room and check the blocks inside
            if PhaseStrong.SEARCH_BLOCK == self._phase:
                self._state_tracker.update(state)
                contents = state.get_room_objects(self._door['room_name'])
                for c in contents:
                    if ("Block" in c['name']) and (c not in self._currentRoomObjects) \
                            and 'GhostBlock' not in c['class_inheritance']:
                        self._currentRoomObjects.append(c)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = PhaseStrong.FOUND_BLOCK
            # if goal blocks are found go to pickup
            if PhaseStrong.FOUND_BLOCK == self._phase:
                self.FOUND_BLOCK_logic(agent_name, self._currentRoomObjects)
                self.FOUND_BLOCK_logic(agent_name, self._currentRoomObjects)
            #picks up the block
            if PhaseStrong.PICKUP_BLOCK == self._phase:
                return self.pickupLogic(agent_name, self._blockToPick, state)
            # go to goal location to drop blocks
            if PhaseStrong.FOLLOW_PATH_TO_GOAL == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goalBlocks[self._currentIndex]['location']])
                self._phase = PhaseStrong.DROP_BLOCK
            # drop the last block you hold and move to search the next one
            if PhaseStrong.DROP_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = PhaseStrong.PLAN_PATH_TO_DOOR
                if len(self._holdingBlocks) > 0:
                    self._currentIndex += 1
                    block = self._holdingBlocks[-1]
                    self._sendMessage(Util.droppingBlockMessage(
                        block, self._goalBlocks[self._currentIndex - 1]['location']), agent_name)
                    return DropObject.__name__, {
                        'object_id': block['obj_id']}
            #go search for blocks above the goal
            if PhaseStrong.PATH_ABOVE_GOAL_DROP == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._drop_location_above_goal])
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = PhaseStrong.CHECK_ABOVE_GOAL_DROP
            #If there are goal blocks you need pick them up
            if PhaseStrong.CHECK_ABOVE_GOAL_DROP == self._phase:
                self._check_above_goal = (self._check_above_goal[0], False)
                agentLocation = state[self.agent_id]['location']
                closeObjects = state.get_objects_in_area((agentLocation[0], agentLocation[1]),
                                                         bottom_right=(agentLocation[0] + 1, agentLocation[1] + 1))
                self._phase = PhaseStrong.PLAN_PATH_TO_DOOR
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
            self._phase = PhaseStrong.PLAN_PATH_TO_DOOR

    ####################################################################################
    ########################### Action Logic ###########################################

    def FOUND_BLOCK_logic(self, agent_name, currentRoomObjects):
        for c in currentRoomObjects:
            if self.isGoalBlock(c):
                self._sendMessage(Util.foundGoalBlockMessage(c), agent_name)
                # manageBlock will decide if the goal block should be picked or not
                self.manageBlock(c)
            else:
                self._sendMessage(Util.foundBlockMessage(c), agent_name)
        action = self._navigator.get_move_action(self._state_tracker)
        if action is not None:
            return action, {}
        self._phase = PhaseStrong.PLAN_PATH_TO_DOOR

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
        self._phase = PhaseStrong.SEARCH_BLOCK
        self._sendMessage(Util.searchingThroughMessage(self._door['room_name']), agent_name)

    def PLAN_PATH_TO_DOOR_logic(self, agent_name):
        self._navigator.reset_full()
        if self._check_above_goal[1]:
            self._phase = PhaseStrong.PATH_ABOVE_GOAL_DROP
            return None, {}

        if self._doorIndex >= len(self._doors):
            # Randomly pick a closed door
            self._door = random.choice(self._doors)
            if self._check_above_goal[0]:
                self._phase = PhaseStrong.PATH_ABOVE_GOAL_DROP
                return None, {}
        self._door = self._doors[self._doorIndex]
        self._doorIndex += 1

        doorLoc = self._door['location']
        # Location in front of door is south from door
        doorLoc = doorLoc[0], doorLoc[1] + 1
        # Send message of current action
        self._sendMessage(Util.moveToMessage(self._door['room_name']), agent_name)
        self._navigator.add_waypoints([doorLoc])
        self._phase = PhaseStrong.FOLLOW_PATH_TO_CLOSED_DOOR

    def pickupLogic(self, agent_name, currentBlock, state):
        self._state_tracker.update(state)
        action = self._navigator.get_move_action(self._state_tracker)
        if action is not None:
            return action, {}
        self._phase = PhaseStrong.PLAN_PATH_TO_DOOR
        goalBlockIndex = self.getGoalBlockIndex(currentBlock)
        #if it is not a goal block go back
        if goalBlockIndex is None:
            return None, {}
        block = self.getGoalBlockName(state, currentBlock)
        #if the block is not at the location go back
        if block is None:
            self._foundGoalBlocks[goalBlockIndex] = None
            return None, {}
        #otherwise pick it up
        self._sendMessage(Util.pickingUpBlockMessage(block), agent_name)
        return GrabObject.__name__, {'object_id': block['obj_id']}

    ################################################################################
    ####################### Block Logic ############################################
    def dropOldGoalBlock(self):
        for block in self._holdingBlocks:
            if self.getOldGoalBlockIndex(block) < self._currentIndex:
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
        reducedGoalBlocks = reducedGoalBlocks[self._currentIndex:]
        try:
            return reducedGoalBlocks.index(blockInfo) + self._currentIndex
        except ValueError:
            return None
    def getOldGoalBlockIndex(self, block):
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
    def setDataAboveGoal(self):
        if self._drop_location_above_goal is None:
            loc = self._goalBlocks[-1]['location']
            self._drop_location_above_goal = (loc[0], loc[1] - 1)
        if self._doorIndex % 2:
            self._check_above_goal = (True, self._check_above_goal[1])
        else:
            self._check_above_goal = (False, self._check_above_goal[1])
    def isGoalBlock(self, block):
        getBlockInfo = lambda x: dict(list(x['visualization'].items())[:3])
        blockInfo = getBlockInfo(block)
        reducedGoalBlocks = [getBlockInfo(x) for x in self._goalBlocks]
        if (blockInfo in reducedGoalBlocks) and not block['is_goal_block'] and not block['is_drop_zone']:
            return True
        return False

    def manageBlock(self, block):
        goalBlockIndex = self.getGoalBlockIndex(block)
        if goalBlockIndex is None:
            return
        self._foundGoalBlocks[goalBlockIndex] = block
        #if I hold the block I do not need to pick it
        if goalBlockIndex in [self.getGoalBlockIndex(x) for x in self._holdingBlocks]:
            return
        # if it is the current block or I have space to pick a later goal block I will pick it
        if goalBlockIndex == self._currentIndex or (
                len(self._holdingBlocks) == 0 and goalBlockIndex > self._currentIndex):
            self._phase = PhaseStrong.PICKUP_BLOCK
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
        #rooms = ['room_1', 'room_0', 'room_3',  'room_2', 'room_4', 'room_5', 'room_6', 'room_7', 'room_8']
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
        shape = 0
        colour = 0
        if "shape" in vis1 and "shape" in vis2:
            shape = 0.05 if vis1['shape'] == vis2['shape'] else -0.05

        if "colour" in vis1 and "colour" in vis2:
            colour = 0.05 if vis1['colour'] == vis2['colour'] else -0.05

        return shape + colour

    ########################################################################
    ################# Update Info From Team ################################
    def badTrustCondition(self, member):
        # if this condition holds, the actions shoud not be taken into account, trust is too low
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
        # if a goal block was found, remember it
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
        if self._foundGoalBlocks[goalBlockIndex] is None:
            return
        #If a goal block you knew about was picked up, forget about it
        if tuple(block['location']) == self._foundGoalBlocks[goalBlockIndex]['location']:
            self._foundGoalBlocks[goalBlockIndex] = None

    def dropBlockUpdate(self, block, member):
        if self.badTrustCondition(member):
            return
        #if the block is dropped above goal search for it
        if tuple(block['location']) == self._drop_location_above_goal:
            self._check_above_goal = (self._check_above_goal[0], True)
        return

    def dropGoalBlockUpdate(self, block, member):
        if self.badTrustCondition(member):
            return
        goalBlockIndex = self.getGoalBlockIndex(block)
        if goalBlockIndex is None:
            return
        #if the block is dropped at goal search for the next blocks
        if self._goalBlocks[goalBlockIndex]['location'] == tuple(block['location']):
            if self._currentIndex == goalBlockIndex:
                self._currentIndex += 1
        else:
            self._foundGoalBlocks[goalBlockIndex] = block
    def pickUpBlockSimpleUpdate(self, block, member):
        return
    def updateRep(self, avg_reps):
        nr_team_mates = len(self._teamMembers)
        for member in avg_reps.keys():
            self._trust[member]['rep'] = (avg_reps[member] + (self._trust[member]['rep'] * (nr_team_mates - 1))) / nr_team_mates