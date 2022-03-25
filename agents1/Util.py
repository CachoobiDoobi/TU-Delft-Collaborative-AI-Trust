import json
import re

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
               + " at location (" + ', '.join([str(loc) for loc in location]) + ")"

    @staticmethod
    def pickingUpBlockMessage(data):
        item_info = dict(list(data['visualization'].items())[:3])
        return "Picking up goal block " + json.dumps(item_info) \
               + " at location (" + ', '.join([str(loc) for loc in data['location']]) + ")"

    @staticmethod
    def droppingBlockMessage(data, location):
        item_info = dict(list(data['visualization'].items())[:3])
        return "Droppped goal block " + json.dumps(item_info) \
               + " at location (" + ', '.join([str(loc) for loc in location]) + ")"

    @staticmethod
    def reputationMessage(trust, team_members):
        rep = {}
        for member in team_members:
            rep[member] = trust[member]['rep']
        return "Reputation:" + json.dumps(rep)

    # TODO -  Implement methods: foundGoalBlockUpdate, foundBlockUpdate, pickUpBlockUpdate, dropBlockUpdate, dropGoalBlockUpdate
    # TODO -  In agent class; Each method takes block & member as arguments (use to update for each agent)

    @staticmethod
    def update_info_general(arrayWorld, receivedMessages, teamMembers,
                            foundGoalBlockUpdate, foundBlockUpdate, pickUpBlockUpdate, dropBlockUpdate, dropGoalBlockUpdate):
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
                        "action": "found",
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
                        "action": "found",
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
                        "action": "found",
                    })