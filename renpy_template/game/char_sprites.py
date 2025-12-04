from math import atan2, cos, pi, sin
from random import uniform

import renpy.exports as renpy
from character import FacingLeft, voice_tomboy
from renpy.character import ADVCharacter
from renpy.display.im import Composite, Flip, Image, MatrixColor, Scale, matrix
from renpy.display.motion import Transform
from renpy.exports import Displayable, Render, error, load_image, scene_lists, store
from renpy.python import RevertableDict, RevertableObject, RevertableSet

#####################
# Classes/Functions #
#####################


def get_sprite_path(root):
    """Returns the path from a CharacterSprite displayable to the root node.
    The returned value is a list with the CharacterSprite as the first element
    and the root as the last, or None if the CharacterSprite was not found.
    """
    if isinstance(root, CharacterSprite):
        return [root]
    for child in root.visit():
        if not child:
            continue
        path = get_sprite_path(child)
        if path is not None:
            path.append(root)
            return path
    else:
        return None


def sprite_of(what, layer="master"):
    if isinstance(what, Person):
        d = scene_lists().get_displayable_by_tag(layer, what.image_tag)
    elif isinstance(what, str):
        d = scene_lists().get_displayable_by_tag(layer, what)
    elif isinstance(what, Transform):
        d = what()
    else:
        renpy.error("Do not know how to get sprite of a %s" % what.__class__.__name__)

    def func(node):
        if not node:
            return None
        if isinstance(node, CharacterSprite):
            return node
        for child in node.visit():
            child_d = func(child)
            if child_d:
                return child_d
        return None

    return func(d)


"""
This intercept is necessary to store the current speaker in a store variable.
This is then accessed in the "say" screen since it does not make
the actual speaker object accessible by default.
"""


def new_adv_call(self, *args, **kwargs):
    self.multiple = kwargs.get("multiple", None)
    renpy.store._speaker = self
    original_adv_call(self, *args, **kwargs)


original_adv_call = ADVCharacter.__call__
ADVCharacter.__call__ = new_adv_call


class VoicedCharacter(ADVCharacter):
    """A speaking character with or without a body.
    Concrete subclasses must provide a property named 'voice' to select the voice this character talks in.
    """

    def __init__(self, name, **kvargs):
        ADVCharacter.__init__(self, name, store.adv, callback=None, **kvargs)

    def voice_beep(self, event, interact=True, **kwargs):
        if event == "show_done":
            renpy.music.play(self.voice, channel="text")
        elif event == "slow_done":
            renpy.music.stop(fadeout=0.3, channel="text")


class Speaker(VoicedCharacter):
    """A speaking character without a body, using a fixed voice."""

    def __init__(self, name, voice, **kvargs):
        self.voice = voice
        VoicedCharacter.__init__(self, name, **kvargs)


class Person(RevertableObject, VoicedCharacter):
    """A character with a body.
    The body can be changed as the game progresses.
    """

    def __init__(self, image_tag, name, body, **kvargs):
        VoicedCharacter.__init__(self, name, **kvargs)
        self.who_args = RevertableDict(self.who_args)

        self.image_tag = image_tag
        self.body = body
        self.name = name

        if body in getattr(renpy.store, "bodies"):
            self.outfit = getattr(renpy.store, "bodies")[body].default_outfit
        else:
            self.outfit = ""

        self.accessories = RevertableSet()

    @property
    def voice(self):
        return getattr(renpy.store, "bodies")[self._body].voice

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, value):
        self._body = value
        bodyObj = getattr(renpy.store, "bodies").get(value)
        self.who_args["color"] = "#ffffff" if bodyObj is None else bodyObj.color

    @property
    def ycenter(self):
        return sprite_of(self).ycenter

    @property
    def zoom(self):
        img = scene_lists().get_displayable_by_tag("master", self.image_tag)
        path = get_sprite_path(img)
        if path is None:
            error(
                'Did not find body graphic for "%s" in its transform tree'
                % self.image_tag
            )
            return 1.0
        else:
            zoom = 1.0
            for d in path[1:]:
                zoom *= getattr(d, "zoom", 1.0)
            return zoom

    @property
    def xzoom(self):
        img = scene_lists().get_displayable_by_tag("master", self.image_tag)
        path = get_sprite_path(img)
        if path is None:
            error(
                'Did not find body graphic for "%s" in its transform tree'
                % self.image_tag
            )
            return 1.0
        else:
            xzoom = 1.0
            for d in path[1:]:
                xzoom *= getattr(d, "xzoom", 1.0)
            return xzoom

    def predict_outfit(self, outfit):
        result = []
        for pose in getattr(renpy.store, "bodies")[self._body].poses.values():
            if outfit in pose.outfits:
                result.append(pose.outfits[outfit])
        return result

    def swap_body(self, other):
        # Swap physical attributes, not name
        self.outfit, other.outfit = other.outfit, self.outfit
        self.accessories, other.accessories = other.accessories, self.accessories
        self.body, other.body = other.body, self.body

    def clone(self, other):
        # Copy physical attributes
        self.outfit = other.outfit
        self.accessories = set(other.accessories)
        self.body = other.body

    def add_accessory(self, accessory):
        if "_" in accessory:
            group = accessory.split("_")[0]
            new_accs = set()
            for item in self.accessories:
                if item.startswith("{}_".format(group)):
                    continue
                new_accs.add(item)
            new_accs.update(
                set([x for x in self.accessories if not x.startswith(group)])
            )
            self.accessories = new_accs
        self.accessories.add(accessory)

    def remove_accessory(self, accessory):
        self.accessories.discard(accessory)

    def clear_accessories(self):
        self.accessories.clear()

    def reset(self):
        self.accessories.clear()
        bodies = getattr(renpy.store, "bodies")
        self.body = bodies[self.image_tag]
        self.outfit = bodies[self.image_tag].default_outfit
        self.name = getattr(renpy.store, "characters")[self.image_tag]


def FakePerson(name, body):
    """A Speaker who has a voice from a body,
    but isn't a fully-fledged person."""
    bodies = renpy.store.bodies
    if body in bodies:
        color = bodies[body].color
        voice = bodies[body].voice
    else:
        color = body
        voice = voice_tomboy
    return Speaker(name, voice, color=color)


class Ghost(Person):
    def __init__(self, image_tag, name, body, host_name, **kvargs):
        Person.__init__(self, image_tag, name, body, **kvargs)

        self.host_name = host_name

        self.velocity = (0.0, 0.0)
        self.pull = (1.0, 0.0)
        self.lastpull = -1000.0
        self.lastanim = 0.0
        self.offset = (0.0, 0.0)

    def exit_host(self):
        # Copy the state of the host if they have the same body,
        # i.e. the original clothing/appearance of the host
        host = getattr(store, self.host_name)
        if self.body == host.body:
            self.outfit = host.outfit
            self.accessories = set(host.accessories)

    def animate(self, trans, st, at):
        dt = at - self.lastanim
        self.lastanim = at
        if dt < 0.0:
            # Going back in time; probably a rollback.
            self.lastpull = -1000.0
            dt = 0.0
        else:
            # If many frames were skipped, consider most of the time as lost.
            dt = min(dt, 0.05)

        ax, ay = self.pull
        if at - self.lastpull > 0.3:
            # Change the pull on the ghost.
            a = atan2(ay, ax)
            if self.lastpull >= 0.0:
                a += pi * uniform(0.2, 1.8)
            r = uniform(50.0, 75.0)
            ax = r * cos(a)
            ay = r * sin(a)
            self.pull = ax, ay
            self.lastpull = at

        vx, vy = self.velocity
        vx += ax * dt
        vy += ay * dt

        xoffset, yoffset = self.offset

        # Pull ghost back to center.
        g = -2.0
        vx += g * xoffset * dt
        vy += g * yoffset * dt

        # Slow down.
        vx *= 1.0 - 0.25 * dt
        vy *= 1.0 - 0.25 * dt

        xoffset += vx * dt
        yoffset += vy * dt

        self.velocity = vx, vy
        self.offset = xoffset, yoffset

        trans.xoffset = xoffset
        trans.yoffset = yoffset
        trans.subpixel = True
        return 0


class GhostFloat(RevertableObject):
    def __init__(self, ghost_name):
        self.ghost_name = ghost_name

    def __call__(self, trans, st, at):
        ghost = getattr(store, self.ghost_name)
        return ghost.animate(trans, st, at)


# This is a displayable used by renpy.image() to render characters with properties
# tracked by Ren'Py rather than Person (i.e. emotion and blush, for now)
class CharacterSprite(Displayable):
    def __init__(self, person_name, emotion, blushing, **kwargs):
        Displayable.__init__(self, **kwargs)

        self.person_name = person_name
        self.emotion = emotion
        self.blushing = blushing

        # Cached info
        self.width = 0
        self.height = 0
        self.img = None
        self.last_blur = 0

    def render(self, width, height, st, at):
        if (
            st == 0.0
            or self.img is None
            or renpy.in_rollback()
            or self.last_blur != store.screenfilter.blur
        ):
            self.width, self.height, self.img = self.compose()
            self.last_blur = store.screenfilter.blur

        if self.img is None:
            return Render(0, 0)

        render = Render(self.width - 1, self.height - 1)
        render.blit(load_image(self.img), (0, 0))

        if renpy.store.coordinate_grid_key_presses:
            try:
                d = renpy.scene_lists().get_displayable_by_tag(
                    "master", self.person_name
                )
                try:
                    tf = renpy.get_placement(d)
                    xanchor = tf.xanchor
                    yanchor = tf.yanchor
                except:
                    xanchor = d.state.inherited_xanchor
                    yanchor = d.state.inherited_yanchor

                point_size = (15, 15)
                dot = Transform(
                    renpy.display.imagelike.Solid("#B22222"), maxsize=point_size
                )
                render.place(
                    dot,
                    x=int(self.width * xanchor) - (point_size[0] // 2),
                    y=int(self.height * yanchor) - (point_size[1] // 2),
                    width=point_size[0],
                    height=point_size[1],
                )

                dot = Transform(
                    renpy.display.imagelike.Solid("#1E90FF"), maxsize=point_size
                )
                render.place(
                    dot,
                    x=(self.width // 2) - (point_size[0] // 2),
                    y=(self.height // 2) - (point_size[1] // 2),
                    width=point_size[0],
                    height=point_size[1],
                )
            except:
                pass

        return render

    def compose(self):
        person, body, pose, images = self._get_objects_and_images(True)
        if person is None:
            return 0, 0, None

        ### START Determine outfit for characters with text or other non-reversible parts on their body ###
        # Always find the base-outfit
        outfit_original = (
            person.outfit[:-9] if "_inverted" in person.outfit else person.outfit
        )
        outfit_inverted = outfit_original + "_inverted"
        # If we are flipped and we have an inverted outfit available, switch to it, otherwise restore the normal outfit
        try:
            person.outfit = (
                outfit_inverted
                if person.xzoom < 0 and outfit_inverted in body.all_outfits
                else outfit_original
            )
        except:
            person.outfit = outfit_original
        # Refresh the data, since we may have changed outfit
        person, body, pose, images = self._get_objects_and_images(True)
        ### END ###

        width, height = body.size[0] * body.scale, body.size[1] * body.scale

        # Compose the images
        composite_args = []
        for img in images:
            composite_args.append(pose.pos)
            composite_args.append(img)
        img = Composite(body.size, *composite_args)

        if pose.direction is FacingLeft:
            img = Flip(img, horizontal=True)
        img = Scale(img, width, height)

        # If screen is blurred (and we're not the protag), blur this image
        blur = store.screenfilter.blur
        if (
            blur > 0.0
            and self.person_name != store.protagonist
            and self.person_name not in renpy.store.phone_images
        ):
            img = renpy.display.im.Blur(img, store.screenfilter.blur / float(12))

        # Return result
        return width, height, img

    def visit(self):
        person, body, pose, images = self._get_objects_and_images(False)
        if body is None:
            return []
        return images

    def per_interact(self):
        renpy.display.render.redraw(self, 0)

    @property
    def size(self):
        person, body = self._get_objects()
        if person is None:
            return None
        return body.size

    @property
    def ycenter(self):
        person, body = self._get_objects()
        if person is None:
            return None
        return body.get_pose(self.emotion).ycenter

    @property
    def pose(self):
        __, __, pose, __ = self._get_objects_and_images(False)
        return pose

    def _get_objects(self):
        person = getattr(store, self.person_name)
        if not isinstance(person, Person):
            return None, None
        body = getattr(renpy.store, "bodies")[person.body.replace("Ghost", "")]
        return person, body

    def _get_objects_and_images(self, fatal):
        person, body = self._get_objects()
        if not person:
            return None, None, None, None
        pose, images = body.get_pose_and_images(
            self.emotion, self.blushing, person, fatal=fatal
        )
        if not pose:
            return None, None, None, None
        images = [store.screenfilter.tint(img) for img in images]
        return person, body, pose, images

    nosave = ["img"]

    def after_setstate(self):
        self.make_dirty()

    def make_dirty(self):
        self.width = 0
        self.height = 0
        self.img = None


# Resources for ghost animations.
red_to_alpha = matrix(0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0)
ghost_mask = MatrixColor(Image("images/ghostmask.png"), red_to_alpha)
ghost_matrix = matrix.saturation(0.0) * matrix.tint(0.7, 0.73, 0.8)


class GhostSprite(CharacterSprite):
    def __init__(self, person_name, emotion, **kwargs):
        CharacterSprite.__init__(self, person_name, emotion, False, **kwargs)

    def visit(self):
        return super(GhostSprite, self).visit() + [ghost_mask]

    def render(self, width, height, st, at):
        # Render ghost-colored body.
        cr = super(GhostSprite, self).render(width, height, st, at)
        size = cr.get_size()

        # Render scaled ghost alpha mask.
        mr = Render(*size)
        mr.blit(load_image(Scale(ghost_mask, *size)), (0, 0))

        # Render null.
        nr = Render(*size)

        # Combine renders.
        rv = Render(*size)

        rv.mesh = True

        rv.add_shader("renpy.imagedissolve")
        rv.add_uniform("u_renpy_dissolve_offset", 0.0)
        rv.add_uniform("u_renpy_dissolve_multiplier", 256.0 / 192.0)

        rv.blit(mr, (0, 0), focus=False, main=False)
        rv.blit(nr, (0, 0), focus=False, main=False)
        rv.blit(cr, (0, 0))

        return rv

    def _get_objects_and_images(self, fatal):
        person, body, pose, images = super(GhostSprite, self)._get_objects_and_images(
            fatal
        )
        if images is not None:
            images = [MatrixColor(img, ghost_matrix) for img in images]
        return person, body, pose, images


# A hash of all sprites
sprites = {}


def define_sprite(name, image):
    renpy.display.image.register_image(name, image)
    sprites[name] = image


def make_sprites_dirty():
    for sprite in sprites.values():
        if isinstance(sprite, Transform):
            sprite = sprite.child
        sprite.make_dirty()
