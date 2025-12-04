import os
from collections import defaultdict

from renpy.display.im import Image
from renpy.exports import error


def qualify(map, outfit, accessories):
    """Returns the map value that qualifies person"""
    key = max(map, key=lambda k: k.match_score(outfit, accessories))
    if key.match_score(outfit, accessories) < 0:
        return None
    return map[key]


class BodyImageQualifier:
    def __init__(self, body, qInfo=None):
        self.outfits = set()
        self.accessories = set()

        if not qInfo:
            return

        if "$" in qInfo:
            self.outfits.add(qInfo["$"])
        if "@" in qInfo:
            for a in qInfo["@"]:
                self.accessories.add(a)
        if "%" in qInfo:
            self.outfits.update(body.mutation_to_outfits[qInfo["%"]])

    def __repr__(self):
        return "BodyImageQualifier(outfits={}, accessories={})".format(
            self.outfits, self.accessories
        )

    def __hash__(self):
        return hash((tuple(self.outfits), tuple(self.accessories)))

    def __eq__(self, other):
        return self.outfits == other.outfits and self.accessories == other.accessories

    def match_score(self, outfit, accessories):
        # Outfit score
        if self.outfits:
            if outfit not in self.outfits:
                return -1
            score = 100
        else:
            score = 0

        # Accessories score
        if self.accessories:
            if not self.accessories.issubset(accessories):
                return -1
            score += len(self.accessories.intersection(accessories))

        return score


class Pose:
    def __init__(self, name, size, center, direction):
        self.name = name
        self.size = size
        self.center = center
        self.direction = direction

        # Graphics
        self.outfits = {}
        self.faces = defaultdict(lambda: defaultdict(dict))
        self.accessories = defaultdict(lambda: defaultdict(dict))

    def init_size(self, w_half, h):
        self.pos = (w_half - self.center[0], h - self.size[1])
        self.ycenter = (self.center[1] + self.pos[1]) / float(h)

    def select_outfit_img(self, outfit):
        # Clothed
        if outfit:
            if outfit in self.outfits:
                return self.outfits[outfit]
            return None
        return None

    def select_face_img(self, name, blushing, outfit, accessories):
        # Get the face image that qualifies
        if name not in self.faces:
            return None
        face = qualify(self.faces[name], outfit, accessories)
        if not face:
            return None

        # Select the most appropriate blush
        if blushing in face:
            return face[blushing]
        if (not blushing) in face:
            return face[not blushing]
        return None

    def select_accessory_imgs(self, outfit, accessories):
        tmp = []
        active_groups = set()
        # Generate a list of (name, accessory) tuples for all accessories that are active
        # At the same time we also generate a set of accessory names that belong to a group and are active
        for accessory_name, qualify_map in self.accessories.items():
            if not qualify_map:
                continue
            # Grab the qualified accessory
            accessory = qualify(qualify_map, outfit, accessories)
            if not accessory:
                continue

            # Get the correct on/off image
            is_on = accessory_name in accessories

            if is_on and "_" in accessory_name:
                active_groups.add(accessory_name.split("_")[0])

            if is_on in accessory:
                tmp.append((accessory_name, accessory[is_on][0], accessory[is_on][1]))

        result = {}
        # We remove all active accessories that are the parent of a group,
        # if at least one of the children of the group is active
        for accessory_name, accessory, zorder in tmp:
            if accessory_name in active_groups:
                continue
            result[accessory_name] = (accessory, zorder)

        return result


class Expression:
    def __init__(self, pose_name, name):
        self.pose_name = pose_name
        self.name = name


# This class holds information about bodies
class Body:
    def __init__(self, color, scale, voice, default_outfit, eye_line, mutations={}):
        self.color = color
        self.scale = scale
        self.voice = voice
        self.default_outfit = default_outfit
        self.eye_line = eye_line
        self.all_outfits = set()
        self._all_accessories = set()
        self.all_expressions = set()
        self.poses = {}
        self.size = 0

        # Parse mutations
        self.mutation_to_outfits = {}
        for mutation_name, outfit_names in mutations.items():
            self.mutation_to_outfits[mutation_name] = frozenset(outfit_names)

    def set_size(self, width, height):
        self.size = (width, height)

    def add_pose(self, pose):
        self.poses[pose.name] = pose

    def add_outfit(self, pose_name, name, filename):
        filename = filename.replace(os.sep, "/")
        self.all_outfits.add(name)
        self.poses[pose_name].outfits[name] = Image(filename)

    def add_face(self, pose_name, name, qualifier, blush, filename):
        filename = filename.replace(os.sep, "/")
        self.all_expressions.add(pose_name + "_" + name)
        self._all_accessories.update(qualifier.accessories)
        self.all_outfits.update(qualifier.outfits)
        self.poses[pose_name].faces[name][qualifier][blush] = Image(filename)

    def add_accessory(self, pose_name, name, qualifier, is_on, filename, zorder=0):
        filename = filename.replace(os.sep, "/")
        self._all_accessories.add(name)
        self._all_accessories.update(qualifier.accessories)
        self.all_outfits.update(qualifier.outfits)
        self.poses[pose_name].accessories[name][qualifier][is_on] = (
            Image(filename),
            zorder,
        )

    @property
    def accessory_groups(self):
        groups = set()
        for acc in self._all_accessories:
            if "_" in acc:
                groups.add(acc.split("_")[0])
        return groups

    @property
    def all_accessories(self):
        groups = self.accessory_groups
        return [acc for acc in self._all_accessories if acc not in groups]

    def emotion_to_expression(self, emotion):
        # pose-expression style
        if emotion.count("_") > 2:
            error("Invalid emotion: {0}".format(emotion))
        emotion_data = emotion.split("_")
        pose_name, name = "_".join(emotion_data[:-1]), emotion_data[-1]
        return Expression(pose_name, name)

    def get_pose(self, emotion):
        expression = self.emotion_to_expression(emotion)
        return self.poses[expression.pose_name]

    def get_pose_and_images(self, emotion, blushing, person, fatal=True):
        expression = self.emotion_to_expression(emotion)
        if expression.pose_name not in self.poses:
            return None, None
        pose = self.poses[expression.pose_name]

        accessories = sorted(
            [
                (image, zorder)
                for image, zorder in pose.select_accessory_imgs(
                    person.outfit, person.accessories
                ).values()
            ],
            key=lambda x: x[1],
        )

        images = []

        images.extend([k for k, v in accessories if v < 0])

        # Outfit/nude base
        img = pose.select_outfit_img(person.outfit)
        if img:
            images.append(img)
        elif fatal:
            error(
                'Outfit "{0}" does not exist in pose "{1}" for body "{2}"'.format(
                    person.outfit, expression.pose_name, person.body
                )
            )

        # Face
        img = pose.select_face_img(
            expression.name, blushing, person.outfit, person.accessories
        )
        if img:
            images.append(img)
        elif fatal:
            error(
                'Face "{0}" does not exist in pose "{1}" for body "{2}"'.format(
                    expression.name, expression.pose_name, person.body
                )
            )

        # Accessories
        images.extend([k for k, v in accessories if v >= 0])

        return pose, images
