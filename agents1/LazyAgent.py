import csv
import enum
import os
import random
from typing import Dict

import numpy as np
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import DropObject
from matrx.actions.object_actions import GrabObject
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message

from agents1.Util import Util
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
    DROP_OBJECT_NEAR_GOAL = 13
    RESET = 14
    PLAN_PATH_TO_DROP_ZONE = 15
    GO_TO_DROP_ZONE = 16


# TODO make the check drop goal near work
class LazyAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.START
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
                                 self.dropBlockUpdate, self.dropGoalBlockUpdate, self.updateRep, self.agent_name)

        while True:
            if Phase.START == self._phase:
                # 'Drop_off_0', 'Drop_off_0_1', 'Drop_off_0_2'
                # {'drop_zone_nr': 0, 'is_drop_zone': True, 'is_goal_block': False, 'is_collectable': False, 'name': 'Drop off 0', 'obj_id': 'Drop_off_0', 'location': (12, 21), 'is_movable': False, 'carried_by': [], 'is_traversable': True,
                # Collect_Block', 'Collect_Block_1', 'Collect_Block_2'
                # {'drop_zone_nr': 0, 'is_drop_zone': False, 'is_goal_block': True, 'is_collectable': False, 'name': 'Collect Block', 'obj_id': 'Collect_Block', 'location': (12, 23), 'is_movable': False, 'carried_by': [], 'is_traversable': True

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

                self.blocks["1"]["delivered"] = False
                self.blocks["2"]["delivered"] = False
                self.blocks["3"]["delivered"] = False

                self.blocks["1"]["locs"] = []
                self.blocks["2"]["locs"] = []
                self.blocks["3"]["locs"] = []

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:

                self._navigator.reset_full()
                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door[
                        'is_open']]

                # Randomly pick a closed door
                if len(closedDoors) != 0:
                    self._door = random.choice(closedDoors)

                else:
                    self._door = random.choice([door for door in state.values()
                                                if 'class_inheritance' in door and 'Door' in door['class_inheritance']])
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
                # -------------------LAZYNESS----------------
                if self.__is_lazy() and self._can_be_lazy:
                    self._phase = random.choice(
                        [Phase.PLAN_PATH_TO_CLOSED_DOOR, Phase.PLAN_PATH_TO_OBJECT, Phase.PLAN_PATH_TO_DROP_ZONE])
                    self._can_be_lazy = False
                # -------------------------------------------

                self._state_tracker.update(state)
                contents = state.get_room_objects(self._door['room_name'])
                contents = list(filter(lambda obj: 'Block' in obj['name'], contents))

                for c in contents:
                    # self._objects.add(c)
                    # self._sendMessage(Util.foundBlockMessage(c), agent_name)

                    for block in self.blocks.values():
                        if self.check_same_visualizations(c['visualization'], block['visualization']) and not c[
                            'is_goal_block'] and not c['is_drop_zone'] and not self.already_delivered(c):

                            if c not in self.foundBlocks:
                                self.foundBlocks.append(c)

                            if block["idx"] == self.current:
                                # print('found')
                                self._phase = Phase.PLAN_PATH_TO_OBJECT
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
                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
                # finished searching room, it can now be lazy again
                self._can_be_lazy = True

            if Phase.PLAN_PATH_TO_OBJECT == self._phase:
                # -------------------LAZYNESS----------------
                if self.__is_lazy() and self._can_be_lazy:
                    self._phase = random.choice([Phase.PLAN_PATH_TO_CLOSED_DOOR, Phase.PLAN_PATH_TO_DROP_ZONE])
                    self._can_be_lazy = False
                # -------------------------------------------
                if len(self.blocks[str(self.current)]['locs']) != 0:
                    self._navigator.reset_full()
                    choice = random.choice(self.blocks[str(self.current)]['locs'])
                    self._navigator.add_waypoints([choice])
                    self.moving_to = choice
                    self._phase = Phase.MOVE_TO_OBJECT
                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.MOVE_TO_OBJECT == self._phase:

                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                self._phase = Phase.GRAB
                self._can_be_lazy = True
            if Phase.GRAB == self._phase:

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
                    self._phase = Phase.MOVING_BLOCK

                    # remove possible location froim dict
                    self.blocks[str(self.current)]['locs'].remove(self.moving_to)
                    self.moving_to = None

                    return GrabObject.__name__, {'object_id': l[0]['obj_id']}

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.MOVING_BLOCK == self._phase:

                # -------------------LAZYNESS----------------
                if self.__is_lazy() and self._can_be_lazy and not self.was_lazy:
                    self._phase = random.choice([Phase.PLAN_PATH_TO_CLOSED_DOOR, Phase.PLAN_PATH_TO_DROP_ZONE])
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

                self._phase = Phase.DROP_OBJECT

            if Phase.DROP_OBJECT == self._phase:
                location_goal = self.blocks["2"]['drop']
                # Get objects in area of location
                objs_in_area = state.get_objects_in_area(location_goal, 1, 1)

                # TODO not working
                for goal in self.blocks.values():
                    for obj in objs_in_area:
                        if self.check_same_visualizations(obj, goal) and obj['location'] == goal['drop']:
                            print("in foist if")
                            if self.current < goal['idx']:
                                print("in second if")
                                self.Phase = Phase.DROP_OBJECT_NEAR_GOAL
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
                    if self.current < 3:
                        self.current += 1
                    else:
                        print("INDEX IS 3 AYYYYYYY")
                    self._phase = Phase.RESET
                    self.was_lazy = False
                    return DropObject.__name__, {'object_id': state[agent_name]['is_carrying'][0]['obj_id']}
                else:
                    if self.current < 3:
                        self.current += 1
                    else:
                        self._phase = Phase.STOP
                    self._phase = Phase.RESET
                    return None, {}

            if Phase.DROP_OBJECT_NEAR_GOAL == self._phase:
                print("drop obj near goal?")
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if action is not None:
                    return action, {}

                self._phase = Phase.RESET
                self._can_be_lazy = True
                self._sendMessage(
                    Util.droppingBlockMessage(self.blocks[str(self.current)], state[agent_name]['location']),
                    agent_name)
                return DropObject.__name__, {'object_id': state[agent_name]['is_carrying'][0]['obj_id']}

            if Phase.RESET == self._phase:

                if len(self.blocks[str(self.current)]['locs']) != 0:
                    self._phase = Phase.PLAN_PATH_TO_OBJECT

                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_PATH_TO_DROP_ZONE == self._phase:
                self._navigator.reset_full()
                loc = self.blocks["3"]['drop']
                as_list = list(loc)
                as_list[1] -= 1
                loc = tuple(as_list)
                self.moving_to = loc
                self._navigator.add_waypoints([loc])
                self._phase = Phase.GO_TO_DROP_ZONE

            if Phase.GO_TO_DROP_ZONE == self._phase:
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
                    self._phase = Phase.MOVING_BLOCK

                    # remove possible location froim dict
                    self.blocks[str(self.current)]['locs'].remove(self.moving_to)
                    self.moving_to = None

                    return GrabObject.__name__, {'object_id': l[0]['obj_id']}

                self.moving_to = None
                self._phase = Phase.RESET

            if Phase.STOP == self._phase:
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
                self._phase = Phase.DROP_OBJECT
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
        for member in self._teamMembers:
            self._trust[member]['rep'] = avg_reps[member] / len(self._teamMembers)
