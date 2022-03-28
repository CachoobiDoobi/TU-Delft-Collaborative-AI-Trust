import enum
import random
from typing import Dict

from matrx.actions.door_actions import OpenDoorAction
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message

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
        self._objects = []
        self._goal_objects_found = []
        self._goal_objects = None
        self._goal_object_delivered = []
        self._current_obj = None

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._searched_doors_index = 0

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
                    self._sendMessage('Moving to room of ' + self._door['room_name'], agent_name)
                else:
                    room_names = [room['room_name'] for room in doors]
                    room_names.remove(self._door['room_name'])
                    self._sendMessage('Moving to room of ' + random.choice(room_names), agent_name)
                self._navigator.add_waypoints([door_location])
                self._phase = Phase.FOLLOW_PATH_TO_DOOR

            if Phase.FOLLOW_PATH_TO_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                if not self._door['is_open']:
                    self._phase = Phase.OPEN_DOOR
                else:
                    self._phase = Phase.SEARCH_ROOM

            if Phase.OPEN_DOOR == self._phase:
                # Send message with a probability of 0.8 to lie
                lie = random.uniform(0, 1)
                if lie <= 0.2:
                    self._sendMessage("Opened door " + self._door['room_name'], agent_name)
                else:
                    door_names = [room['room_name'] for room in [door for door in state.values()
                                                                 if 'class_inheritance' in door and 'Door' in door[
                                                                     'class_inheritance']]]
                    door_names.remove(self._door['room_name'])
                    self._sendMessage("Opened door " + random.choice(door_names), agent_name)

                self._phase = Phase.SEARCH_ROOM
                # Open door
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.SEARCH_ROOM == self._phase:
                self._navigator.reset_full()
                room_area = []
                for area in state.get_room_objects(self._door['room_name']):
                    if "wall" not in area['name']:
                        room_area.append((area["location"][0], area["location"][1]))

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
                                # print(self._door['room_name'])
                                # print(c)
                                self._phase = Phase.MOVE_TO_OBJECT
                                self._current_obj = c
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([c['location']])
                                action = self._navigator.get_move_action(self._state_tracker)
                                return action, {}

                    if "Block" in c['name'] and goal is False:
                        self._objects.append(c)
                        # Send message with a probability of 0.8 to lie
                        lie = random.uniform(0, 1)
                        message = "Found block {\"size\": " + str(c['visualization']['size']) + ", \"shape\": " + \
                                  str(c['visualization']['shape']) + "} at location " + str(c['location'])
                        # if lie <= 0.8:
                        #     size =
                        #     message = "Found block {\"size\": " + str(c['visualization']['size']) +
                        #     ", \"shape\": " + \
                        #         str(c['visualization']['shape']) + "} at location " + str(c['location'])
                        self._sendMessage(message, agent_name)

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
                # state.get_world_info()
                return GrabObject.__name__, {'object_id': self._current_obj['obj_id']}


            if Phase.GRAB == self._phase:
                # print("ggrabbing")
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goal_objects[0]['location']])
                self._phase = Phase.MOVING_BLOCK

            if Phase.MOVING_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                #print("moving to block")

                if action is not None:
                    return action, {}

                # if len(self._goal_objects) == 0:
                #     self._phase = Phase.STOP
                if state[agent_name]['is_carrying']:
                    # print("dropppgbg")
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
