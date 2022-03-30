import enum
import random
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
        self._door = None
        self._objects = []  # All objects found
        self._goal_blocks = []  # Universally known goal blocks
        self._goal_blocks_locations = []  # Goal block locations as said by other agents
        self._goal_blocks_locations_followed = []  # Goal block locations already followed (by blind)
        self._goal_blocks_location_followed_by_others = []  # Goal block locations where other agents went
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

    # Remove colour from any block visualization
    def filter_bw4t_observations(self, state):
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
                    # If another agent has opened the door in the meantime, go search room & send message
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
                    if loc['location'] not in self._goal_blocks_locations_followed:
                        follow = loc['location']
                        self._goal_blocks_locations_followed.append(follow)
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
                location_goal = self._goal_blocks_locations_followed[-1]
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
                l = list(filter(lambda obj: 'Block' in obj['name'] and obj['location'] == location_to_check_followed,
                                objs_in_area))
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
        self.removeLocationFollowedByOther(block)

    def pickUpBlockSimpleUpdate(self, block, member):
        self.removeLocationFollowedByOther(block)

    def removeLocationFollowedByOther(self, block):
        location_goal = block['location']
        for loc in self._goal_blocks_locations:
            if loc['location'] == location_goal:
                self._goal_blocks_locations.remove(loc)
                if loc not in self._goal_blocks_location_followed_by_others:
                    self._goal_blocks_location_followed_by_others.append(loc)

        # Update trust for all objs in goal_blocks_location_followed_by_others
        for obj in self._goal_blocks_location_followed_by_others:
            member = obj['member']
            trust = self._trust[member]['pick-up'] if self._trust[member]['verified'] > 2 else self._trust[member]['rep']
            obj['trustLevel'] = trust

        # Sort locations followed ascending on trust level
        # These locations are going to be checked, so go ascending on trust level (first check least trusted agent)
        self._goal_blocks_location_followed_by_others.sort(key=lambda x: x['trustLevel'])

    def foundBlockUpdate(self, block, member):
        return

    def dropBlockUpdate(self, block, member):
        return

    def dropGoalBlockUpdate(self, block, member):
        return

    def updateRep(self, avg_reps):
        nr_team_mates = len(self._teamMembers)
        for member in avg_reps.keys():
            self._trust[member]['rep'] = (avg_reps[member] + (
                        self._trust[member]['rep'] * (nr_team_mates - 1))) / nr_team_mates

    def updateGoalBlocks(self, state):
        if len(self._goal_blocks) == 0:
            self._goal_blocks = [goal for goal in state.values()
                        if 'is_goal_block' in goal and goal['is_goal_block']]

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
        self._trustBelief(state, closeBlocks)

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
        shape = 0
        colour = 0

        if "shape" in vis1 and "shape" in vis2:
            shape = 0.05 if vis1['shape'] == vis2['shape'] else -0.05

        if "colour" in vis1 and "colour" in vis2:
            colour = 0.05 if vis1['colour'] == vis2['colour'] else -0.05

        return shape + colour
