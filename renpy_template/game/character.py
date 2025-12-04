import os

import renpy
import yaml
from body import Body, BodyImageQualifier, Pose
from pymage_size import get_image_size
from renpy.object import Sentinel

FacingLeft = Sentinel("FacingLeft")
FacingRight = Sentinel("FacingRight")

# Voices (generic character voices)
voice_deep = "e3"
voice_male = "g3"
voice_tomboy = "g4"
voice_woman = "a#4"
voice_girl = "c5"
voice_child = "e5"


#####################
# Classes/Functions #
#####################


class UnknownVoiceException(Exception):
    pass


def add_to_dict(mapping, path, value):
    if not mapping[path[0]]["content"]:
        mapping[path[0]]["content"] = []
    if len(path) == 1:
        if value.endswith(tuple(renpy.store.allowed_image_formats_character)):
            mapping[path[0]]["content"].append(value)
        return mapping
    return add_to_dict(mapping[path[0]], path[1:], value)


def parse_character_structure(tree_map, filename, native=False):
    path = filename.split(os.sep if native else "/")
    if len(path) == 1:
        if not tree_map["content"]:
            tree_map["content"] = []
        tree_map["content"].append(path[0])
    else:
        add_to_dict(tree_map, path[:-1], path[-1])


def list_files(mapping):
    # List the files in a given location of the mapping.
    return mapping["content"]


def list_dirs(mapping):
    # List the directories in a given location of the mapping.
    dirs = list(mapping.keys())
    if "content" in dirs:
        dirs.remove("content")
    return dirs


def add_external_character(char_dir, prefix=""):
    # loads chars from filesystem
    pass


def add_local_character(char_tree_map, char_name, path_prefix, prefix=""):
    # loads chars from interal resources (i.e. filesystem, apk, rpa etc)
    if not list_dirs(char_tree_map):
        renpy.store.logger.warn(
            "This character directory is empty: {}".format(char_name)
        )
        return

    try:
        yaml_file = "/".join(
            [path_prefix.replace("\\", "/"), char_name, "character.yml"]
        )
        with renpy.exports.file(yaml_file) as f:
            char_data = yaml.full_load(f)
    except Exception as e:
        renpy.store.logger.warn(
            "Character definition does not exist or could not be loaded for character"
            " '{}', initializing without: {}".format(char_name, e)
        )
        char_data = {}

    if not prefix:
        code_name = char_name
    else:
        code_name = "{}_{}".format(prefix, char_name)

    outfits = [
        os.path.splitext(item)[0] for item in list_files(char_tree_map["a"]["outfits"])
    ]
    outfits += list_dirs(char_tree_map["a"]["outfits"])
    default_outfit = None
    if "uniform" in outfits:
        default_outfit = "uniform"
    elif "casual" in outfits:
        default_outfit = "casual"
    elif "nude" in outfits:
        default_outfit = "nude"
    else:
        default_outfit = outfits[0]

    body = Body(
        color=char_data.get("name_color", "#ffffff"),
        scale=char_data.get("scale", 1.0),
        voice=get_voice(char_data.get("voice", "girl")),
        default_outfit=char_data.get("default_outfit", default_outfit),
        eye_line=char_data.get("eye_line", 0.0),
        mutations=char_data.get("mutations", {}),
    )

    bodies = getattr(renpy.store, "bodies")
    characters = getattr(renpy.store, "characters")
    all_emotions = getattr(renpy.store, "all_emotions")
    all_outfits = getattr(renpy.store, "all_outfits")
    bodies[code_name] = body
    characters[code_name] = char_data.get("display_name", char_name.capitalize())

    # Pose initialisation
    poses = []
    for pose_name in list_dirs(char_tree_map):
        pose = create_pose(
            char_tree_map[pose_name],
            path_prefix,
            char_name,
            pose_name,
            char_data.get("poses", {}),
        )
        if not pose:
            continue
        poses.append(pose)

    max_w_half, max_h = 0, 0
    for pose in [pose for pose in poses]:
        w_left = pose.center[0]
        w_right = pose.size[0] - w_left
        if w_left > max_w_half:
            max_w_half = w_left
        if w_right > max_w_half:
            max_w_half = w_right
        if pose.size[1] > max_h:
            max_h = pose.size[1]

    body.set_size(max_w_half * 2, max_h)

    for pose in poses:
        pose.init_size(max_w_half, max_h)
        body.add_pose(pose)

    for pose_item in list_dirs(char_tree_map):
        #############
        # Outfits   #
        #############

        # simple outfits
        for outfit_item in list_files(char_tree_map[pose_item]["outfits"]):
            name, ext = os.path.splitext(os.path.basename(outfit_item))
            outfit_path = "/".join(
                [path_prefix, char_name, pose_item, "outfits", outfit_item]
            )
            body.add_outfit(pose_item, name, outfit_path)

        global_accessories = []

        # accessorised outfits
        for outfit_item in list_dirs(char_tree_map[pose_item]["outfits"]):
            if outfit_item.startswith("acc_"):
                global_accessories.append(outfit_item)
                continue

            outfit_path = "/".join(
                [
                    path_prefix,
                    char_name,
                    pose_item,
                    "outfits",
                    outfit_item,
                    list_files(char_tree_map[pose_item]["outfits"][outfit_item])[0],
                ]
            )
            body.add_outfit(pose_item, outfit_item, outfit_path)
            for accessory_item in list_dirs(
                char_tree_map[pose_item]["outfits"][outfit_item]
            ):
                accessory_path = "/".join(
                    [
                        path_prefix.replace("\\", "/"),
                        char_name,
                        pose_item,
                        "outfits",
                        outfit_item,
                        accessory_item,
                    ]
                )

                zorder = 0
                accessory_item_raw = accessory_item
                if accessory_item[-2] in ("+", "-"):
                    zorder = int(accessory_item[-2:])
                    accessory_item = accessory_item[:-1].rstrip("+-")

                qualifier = BodyImageQualifier(body, {"$": outfit_item})

                # Loop through all files in the outfit directory
                # If a file begins with "on_", it belongs to an accessory group
                # This means we register it as "<accessory_name>_<filename>"
                # This is later used to conditionally turn on or off members of the group when necessary
                for accessory_item_state in list_files(
                    char_tree_map[pose_item]["outfits"][outfit_item][accessory_item_raw]
                ):
                    on_img = renpy.store.image_path(
                        "/".join([accessory_path, accessory_item_state])
                    )
                    body.add_accessory(
                        pose_item,
                        (
                            accessory_item
                            if os.path.splitext(accessory_item_state)[0]
                            in ("on", "off")
                            else "{}_{}".format(
                                accessory_item,
                                os.path.splitext(accessory_item_state[3:])[0],
                            )
                        ),
                        qualifier,
                        (
                            False
                            if os.path.splitext(accessory_item_state)[0] == "off"
                            else True
                        ),
                        on_img,
                        zorder,
                    )

                on_img = renpy.store.image_path("/".join([accessory_path, "on"]))
                if on_img:
                    if renpy.exports.loadable(on_img) or os.path.isfile(on_img):
                        body.add_accessory(
                            pose_item, accessory_item, qualifier, True, on_img, zorder
                        )

                off_img = renpy.store.image_path("/".join([accessory_path, "off"]))
                if off_img:
                    if renpy.exports.loadable(off_img) or os.path.isfile(off_img):
                        body.add_accessory(
                            pose_item, accessory_item, qualifier, False, off_img, zorder
                        )

        # global accessories per pose
        for accessory_item in global_accessories:
            accessory_path = "/".join(
                [
                    path_prefix.replace("\\", "/"),
                    char_name,
                    pose_item,
                    "outfits",
                    accessory_item,
                ]
            )
            accessory_name = accessory_item.replace("acc_", "")

            zorder = 0
            accessory_item_raw = accessory_item
            if accessory_name[-2] in ("+", "-"):
                zorder = int(accessory_name[-2:])
                accessory_name = accessory_name[:-1].rstrip("+-")

            for outfit_item in body.poses[pose_item].outfits:
                c_poses = char_data.get("poses", {})
                c_pose_item = c_poses.get(pose_item, {})
                c_pose_excludes = c_pose_item.get("excludes", {})
                c_acc_excludes = c_pose_excludes.get(accessory_name, [])

                if outfit_item in c_acc_excludes:
                    renpy.store.logger.info(
                        "Skipping full accessory {} for {}".format(
                            accessory_name, outfit_item
                        )
                    )
                    continue

                if any(
                    outfit_item in qualifier.outfits
                    for qualifier in body.poses[pose_item].accessories[accessory_name]
                ):
                    # if the given outfit already has an outfit-level accessory with the same name
                    # we don't touch it and ignore the global (less specific) accessory.
                    renpy.store.logger.info(
                        "Skipping full accessory {} because more specific outfit"
                        " accessory was found for pose {}".format(
                            accessory_item, pose_item
                        )
                    )
                    continue

                qualifier = BodyImageQualifier(body, {"$": outfit_item})

                # Loop through all files in the outfit directory
                # If a file begins with "on_", it belongs to an accessory group
                # This means we register it as "<accessory_name>_<filename>"
                # This is later used to conditionally turn on or off members of the group when necessary
                for accessory_item_state in list_files(
                    char_tree_map[pose_item]["outfits"][accessory_item_raw]
                ):
                    on_img = renpy.store.image_path(
                        "/".join([accessory_path, accessory_item_state])
                    )
                    acc_name = (
                        accessory_name
                        if os.path.splitext(accessory_item_state)[0] in ("on", "off")
                        else "{}_{}".format(
                            accessory_name,
                            os.path.splitext(accessory_item_state[3:])[0],
                        )
                    )
                    c_acc_excludes = c_pose_excludes.get(acc_name, {})
                    if outfit_item in c_acc_excludes:
                        renpy.store.logger.info(
                            "Skipping partial accessory {} for {}".format(
                                acc_name, outfit_item
                            )
                        )
                        continue
                    body.add_accessory(
                        pose_item, acc_name, qualifier, True, on_img, zorder
                    )

                on_img = renpy.store.image_path("/".join([accessory_path, "on"]))
                if on_img:
                    if renpy.exports.loadable(on_img) or os.path.isfile(on_img):
                        body.add_accessory(
                            pose_item, accessory_name, qualifier, True, on_img, zorder
                        )

                off_img = renpy.store.image_path("/".join([accessory_path, "off"]))
                if off_img:
                    if renpy.exports.loadable(off_img) or os.path.isfile(off_img):
                        body.add_accessory(
                            pose_item, accessory_name, qualifier, False, off_img, zorder
                        )

        #############
        # Faces     #
        #############

        add_faces(
            body,
            char_tree_map[pose_item]["faces"]["face"],
            path_prefix,
            char_name,
            pose_item,
            False,
        )
        add_accessories(
            body,
            char_tree_map[pose_item]["faces"]["face"],
            path_prefix,
            char_name,
            pose_item,
            False,
        )

        #############
        # Blushes   #
        #############

        # TODO: broken
        if "blush" in char_tree_map[pose_item]["faces"]:
            add_faces(
                body,
                char_tree_map[pose_item]["faces"]["blush"],
                path_prefix,
                char_name,
                pose_item,
                True,
            )
            add_accessories(
                body,
                char_tree_map[pose_item]["faces"]["blush"],
                path_prefix,
                char_name,
                pose_item,
                True,
            )

        #############
        # Mutations #
        #############

        if "mutations" in char_tree_map[pose_item]["faces"]:
            for mutation_item in list_dirs(
                char_tree_map[pose_item]["faces"]["mutations"]
            ):
                add_faces(
                    body,
                    char_tree_map[pose_item]["faces"]["mutations"][mutation_item][
                        "face"
                    ],
                    path_prefix,
                    char_name,
                    pose_item,
                    False,
                    {"%": mutation_item},
                    mutation_item,
                )
                add_accessories(
                    body,
                    char_tree_map[pose_item]["faces"]["mutations"][mutation_item][
                        "face"
                    ],
                    path_prefix,
                    char_name,
                    pose_item,
                    False,
                    {"%": mutation_item},
                    mutation_item,
                )

                add_faces(
                    body,
                    char_tree_map[pose_item]["faces"]["mutations"][mutation_item][
                        "blush"
                    ],
                    path_prefix,
                    char_name,
                    pose_item,
                    True,
                    {"%": mutation_item},
                    mutation_item,
                )
                add_accessories(
                    body,
                    char_tree_map[pose_item]["faces"]["mutations"][mutation_item][
                        "blush"
                    ],
                    path_prefix,
                    char_name,
                    pose_item,
                    True,
                    {"%": mutation_item},
                    mutation_item,
                )

    for pose in poses:
        for outfit in pose.outfits.keys():
            all_outfits.add(outfit)
        for face_name in pose.faces.keys():
            all_emotions.add("{}_{}".format(pose.name, face_name))

    renpy.store.logger.debug("Successfully loaded character '{}'".format(char_name))


def get_rel_path(path, common_prefix):
    return os.path.relpath(path, common_prefix)


def add_faces(
    body,
    face_tree_map,
    path_prefix,
    char_name,
    pose_item,
    blush,
    qualifierType=None,
    mutation_item=None,
):
    qualifier = BodyImageQualifier(body, qualifierType)

    # If we're a file then it's a face item
    for face_item in list_files(face_tree_map):
        if mutation_item:
            face_path = "/".join(
                [
                    path_prefix,
                    char_name,
                    pose_item,
                    "faces",
                    "mutations",
                    mutation_item,
                    "blush" if blush else "face",
                    face_item,
                ]
            )
        else:
            face_path = "/".join(
                [
                    path_prefix,
                    char_name,
                    pose_item,
                    "faces",
                    "blush" if blush else "face",
                    face_item,
                ]
            )
        name, ext = os.path.splitext(face_item)
        body.add_face(pose_item, name, qualifier, blush, face_path)


def add_accessories(
    body,
    face_tree_map,
    path_prefix,
    char_name,
    pose_item,
    blush,
    qualifierType=None,
    mutation_item=None,
):
    for accessory in list_dirs(face_tree_map):
        if mutation_item:
            accessory_path = "/".join(
                [
                    path_prefix,
                    char_name,
                    pose_item,
                    "faces",
                    "mutations",
                    mutation_item,
                    "blush" if blush else "face",
                    accessory,
                ]
            )
        else:
            qualifierType = {}
            accessory_path = "/".join(
                [
                    path_prefix,
                    char_name,
                    pose_item,
                    "faces",
                    "blush" if blush else "face",
                    accessory,
                ]
            )

        accessory_name = os.path.split(accessory_path)[1]
        qualifierType["@"] = (
            accessory_name.split()
        )  # add the accessory qualifier to the qualifierType
        qualifier = BodyImageQualifier(body, qualifierType)

        for accessory_item in list_files(face_tree_map[accessory]):
            name, ext = os.path.splitext(accessory_item)
            final_accessory_path = "/".join([accessory_path, accessory_item])
            body.add_face(pose_item, name, qualifier, blush, final_accessory_path)


def get_voice(voice_name):
    if voice_name == "girl":
        return voice_girl
    elif voice_name == "woman":
        return voice_woman
    elif voice_name == "male":
        return voice_male
    elif voice_name == "child":
        return voice_child
    elif voice_name == "tomboy":
        return voice_tomboy
    elif voice_name == "deep":
        return voice_deep
    raise UnknownVoiceException(
        "Can't parse '{}' into a known voice".format(voice_name)
    )


def create_pose(pose_tree_map, path_prefix, char_name, pose_name, poses_data):
    outfits = pose_tree_map["outfits"]["content"]
    for outfit_name in list_dirs(pose_tree_map["outfits"]):
        for item in pose_tree_map["outfits"][outfit_name]["content"]:
            # find all images that are not named "on" or "off" (to exclude accessories)
            if not item.startswith(("on", "off")):
                outfits.append("/".join([outfit_name, item]))

    # get the size of the first image and use that
    if not outfits:
        raise Exception(
            "The outfits directory of '{}' does not exist or contains no valid outfits".format(
                char_name
            )
        )

    try:
        filename = "/".join([path_prefix, char_name, pose_name, "outfits", outfits[0]])
        try:
            f = renpy.exports.file(filename.replace("\\", "/"))
        except:
            f = open(filename.replace("\\", "/"))
        width, height = get_image_size(f).get_dimensions()
        f.close()
    except:
        raise Exception(
            "Encountered an unknown image format for file '{}' for character '{}' in"
            " pose '{}'".format(outfits[0], char_name, pose_name)
        )
    center_x, center_y = width // 2, height // 2

    facingString = "left"
    pose_data = poses_data.get(pose_name, {})
    if pose_data:
        width = pose_data.get("image_width", width)
        height = pose_data.get("image_height", height)
        center_x = pose_data.get("center_width", width // 2)
        center_y = pose_data.get("center_height", height // 2)
        facingString = pose_data.get("facing", facingString)
    if facingString == "left":
        facing = FacingLeft
    elif facingString == "right":
        facing = FacingRight
    else:
        raise ValueError("Unknown facing direction '{}'".format(facingString))

    return Pose(pose_name, (width, height), (center_x, center_y), facing)
