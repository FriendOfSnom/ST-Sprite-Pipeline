init -100 python:
    # General utility functions.
    def direct_path(image_name):
        sprite = ImageReference(image_name)._target()
        if isinstance(sprite, FilteredImage):
            return sprite.child.filename
        elif isinstance(sprite, Movie):
            return sprite._play
        else:
            return sprite.filename

    def placement_of(what, layer='master'):
        if isinstance(what, Person):
            d = renpy.scene_lists().get_displayable_by_tag(layer, what.image_tag)
        elif isinstance(what, Transform):
            d = what()
        elif isinstance(what, basestring):
            d = renpy.scene_lists().get_displayable_by_tag(layer, what)
        else:
            renpy.error('Do not know how to get placement of a %s' % what.__class__.__name__)
        try:
            placement = renpy.get_placement(d)
        except:
            placement = {}
        # Patch up unknown properties to at least be valid numbers.
        for attr in ('xanchor', 'yanchor', 'xpos', 'ypos'):
            if getattr(placement, attr, None) is None:
                setattr(placement, attr, 0.0)
        # In Ren'Py 8.2, get_placement returns a placement object, which in turn returns position objects.
        # the latter have a method 'simplify' which converts it back to the format we expect here.
        for attr in ("xpos", "ypos", "xanchor", "yanchor", "xoffset", "yoffset", "subpixel"):
            if hasattr(placement, attr):
                attr_obj = getattr(placement, attr)
                if isinstance(attr_obj, renpy.atl.position):
                    setattr(placement, attr, attr_obj.simplify())
        return placement

    def transform_of(what, layer='master'):
        if isinstance(what, Person):
            try:
                placement = renpy.get_placement(renpy.scene_lists().get_displayable_by_tag(layer, what.image_tag))
            except:
                return Transform()
            return Transform(xpos=placement.xpos,
                xanchor=placement.xanchor,
                xoffset=placement.xoffset,
                ypos=placement.ypos,
                yanchor=placement.yanchor,
                yoffset=placement.yoffset,
                xzoom=what.xzoom,
                zoom=what.zoom)
        elif isinstance(what, Transform):
            return what
        else:
            renpy.error('Do not know how to get transform of a %s' % what.__class__.__name__)

    # Image filters.

    def silhouette_matrix(r, g, b, a=1.0):
        return im.matrix((0, 0, 0, 0, r,
                        0, 0, 0, 0, g,
                        0, 0, 0, 0, b,
                        0, 0, 0, a, 0,))

    def silhouetted(image, r, g, b, a=1.0):
        return im.MatrixColor(image, silhouette_matrix(r,g,b,a))

    def create_silhouette(sprite_name, r=0, g=0, b=0, a=1, dynamic_name=None):
        sprite = ImageReference(sprite_name)._target()
        __, height, old_image = sprite.compose()
        silhouette = silhouetted(old_image, r, g, b, a)
        if dynamic_name:
            renpy.display.image.register_image(tuple(dynamic_name.split(" ")), silhouette)
        return silhouette

    # Returns a color matrix that produces a colder-looking version of an image.
    # Amount goes from 0 (unchanged) to 1 (full).
    def colder(amount):
        return (
            im.matrix.brightness(-0.125 * amount) *
            im.matrix.contrast(1.0 - 0.25 * amount) *
            im.matrix.saturation(1.0 - amount)
            )

    # Custom displayables.
    from collections import OrderedDict

    def rectangle_displayable(color=(255, 255, 255, 255), width=1, height=1):
        """ Create a displayable to use as a particle.

        Args:
            colour (tuple): RGBA for the particle.
            width (int): Width of the particle.
            height (int): Height of the particle.

        Returns:
            displayable
        """
        star_color = Solid(color)
        return Fixed(star_color, xysize=(width, height))

    def FallingParticles(image, max_particles=50, speed=150, wind=100, xborder=(0,100), yborder=(50,400), **kwargs):
        return Particles(FallingParticleFactory(image, max_particles, speed, wind, xborder, yborder, **kwargs))

    class FallingParticleFactory(NoRollback):

        def __init__(self, image, max_particles, speed, wind, xborder, yborder, **kwargs):
            self.max_particles = max_particles
            self.speed = speed
            self.wind = wind
            self.xborder = xborder
            self.yborder = yborder
            self.depth = kwargs.get("depth", 10)
            self.image = self.image_init(image)

        def create(self, particles, st):
            if len(particles) < self.max_particles:
                parts = []
                for i in range(max(self.max_particles - len(particles), 0)):
                    depth = random.randint(1, self.depth)
                    depth_speed = 1.5 - depth / float(self.depth)
                    parts.append(renpy.display.particle.SnowBlossomParticle(self.image[depth - 1],
                                                random.uniform(-self.wind, self.wind) * depth_speed,
                                                self.speed * depth_speed,
                                                50,
                                                st + random.randint(0, int(-0.05 * self.speed + 93)),
                                                random.uniform(0, 100),
                                                fast=False,
                                                rotate=False))
                return parts

        def image_init(self, image):
            rv = []
            for depth in range(self.depth):
                p = 1.1 - depth / float(self.depth)
                if p > 1:
                    p = 1.0
                rv.append(im.FactorScale(im.Alpha(image, p), p))

            return rv

        def predict(self):
            return self.image

    class FallingParticle(NoRollback):

        def __init__(self, image, wind, speed, xborder, yborder):
            self.image = image
            if speed <= 0:
                speed = 1
            self.wind = wind
            self.speed = speed
            self.oldst = None
            self.xpos = random.uniform(0-xborder, renpy.config.screen_width+xborder)
            self.ypos = -yborder

        def update(self, st):
            if self.oldst is None:
                self.oldst = st

            lag = st - self.oldst
            self.oldst = st

            self.xpos += lag * self.wind
            self.ypos += lag * self.speed

            if self.ypos > renpy.config.screen_height or \
                (self.wind< 0 and self.xpos < 0) or \
                (self.wind > 0 and self.xpos > renpy.config.screen_width):
                return None

            return int(self.xpos), int(self.ypos), st, self.image

    class Starfield(renpy.Displayable, NoRollback):

        def __init__(self, star_amount=128, star_color=(255, 255, 255, 255), depth=8, speed=0.4, image=None):
            super(renpy.Displayable, self).__init__()
            self.depth = depth
            self.speed = speed
            self.star_amount = star_amount
            self.image = renpy.easy_displayable(image or rectangle_displayable(color=star_color, width=1, height=1))
            self.stars = [self.star_data() for x in range(self.star_amount)]
            self.oldst = None

        def star_data(self):
            alpha = random.uniform(0.2, 1)
            return [renpy.random.randrange(0, config.screen_width),
                    renpy.random.randrange(0, config.screen_height),
                    renpy.random.randrange(1, self.depth),
                    alpha, "up" if alpha < 0.5 else "down",
                    renpy.random.randrange(1, 2),
                    renpy.random.randrange(1, 4)]

        def render(self, width, height, st, at):
            if self.oldst is None:
                self.oldst = st

            lag = st - self.oldst
            self.oldst = st

            render = renpy.Render(width, height)

            move = abs(lag * self.speed)

            for star in self.stars:
                star[5] -= move
                star[6] -= move

                if star[4] == "down":
                    star[3] -= move
                else:
                    star[3] += move

                if star[4] == "up" and star[3] >= 1:
                    if star[5] <= 0:
                        star[4] = "down"
                        star[5] = renpy.random.randrange(1, 2)
                elif star[4] == "down" and star[3] <= 0.2:
                    if star[6] <= 0:
                        star[4] = "up"
                        star[6] = renpy.random.randrange(1, 4)

                transform = Transform(child=self.image, zoom=star[2], alpha=star[3])
                render.place(transform, star[0], star[1])

            renpy.redraw(self, 0.1)

            return render

    # See this: https://github.com/jsfehler/renpy-projection-starfield
    class ProjectionStarfield(renpy.Displayable, NoRollback):
        """ Fires a displayable from the centre of the screen outwards.

        Args:
            star_amount (int): Number of stars to display
            depth (int): Highest z coordinate
            perspective (float): Amount of perspective projection to use
            speed (int): How quickly the stars move off screen
            image (displayable): Visual representation of a star.

        Attributes:
            origin_x (int): Center x coordinate of the projection.
            origin_y (int): Center y coordinate of the projection.
            ranges (list): xy coordinates where a particle can spawn.
            depth_ranges (list): Every possible depth. Spawning randomly picks from this list.
            transforms_amount (int): Amount of size/alpha transformations
        """
        def __init__(self, star_amount=128, depth=16, perspective=128.0, speed=5, spawn_area=25, image=None):
            super(renpy.Displayable, self).__init__()

            self.star_amount = star_amount
            self.depth = depth
            self.perspective = perspective
            self.speed = speed

            if image is None:
                image = rectangle_displayable(width=1, height=1)

            self.image = renpy.easy_displayable(image)

            self.origin_x = config.screen_width * 0.5
            self.origin_y = config.screen_height * 0.5

            self.transforms = self.__precalculate_transforms(self.image)
            self.transforms_amount = len(self.transforms) - 1

            self.stars = [self.__star_data() for x in range(self.star_amount)]

            self.ranges = range(-spawn_area, spawn_area)
            self.ranges.remove(0) # Spawning in the center of the screen is ugly

            self.depth_ranges = range(1, self.depth)

            self.oldst = None

        def __star_data(self):
            """ Create a list with data for each star.

            ie: [x, y, depth, transform index]

            Returns:
                list: Coordinates for building a star
            """
            lower = -25
            higher = 25

            return [
                renpy.random.randrange(lower, higher),
                renpy.random.randrange(lower, higher),
                renpy.random.randrange(1, self.depth),
                renpy.random.randrange(0, self.transforms_amount)
            ]

        def __precalculate_transforms(self, image):
            """
            Pre-calculate all the size/alpha transforms that are possible
            so they don't have to be recreated in render() every single frame.

            Returns:
                list: Displayables for every possible size/alpha
            """
            # All possible depths. Start from maximum depth and decrease
            current_depth = float(self.depth)
            all_depths = []
            step = 0.09

            while current_depth > 0:
                all_depths.append(current_depth)
                current_depth -= step

            # All possible transform factors
            # Using Linear Interpolation, make distant stars smaller and darker than closer stars.
            t_factors = [(1 - float(d) / self.depth) * 2 for d in all_depths]

            # Create list with a Transform() for every possible star
            return [(Transform(child=image, zoom=item, alpha=item)) for item in t_factors]

        def visit(self):
            return self.transforms

        def render(self, width, height, st, at):
            if self.oldst is None:
                self.oldst = st

            lag = st - self.oldst
            self.oldst = st

            render = renpy.Render(0, 0)
            place = render.place
            choice = renpy.random.choice

            w = config.screen_width
            h = config.screen_height

            move = abs(lag * self.speed)

            ranges = self.ranges

            for star in self.stars:
                # Z coordinate decreases each redraw, bringing it closer to the viewer.
                star[2] -= move

                # Star becomes bigger and more visible the further down the list it goes
                star[3] += 1

                # The star is at maximum size/brightness, stop increasing the index
                star[3] = min(star[3], self.transforms_amount)

                # If the star hits zero depth, move it to the back of the projection with random X and Y coordinates.
                if star[2] <= 0:
                    star[0] = choice(ranges)
                    star[1] = choice(ranges)
                    star[2] = choice(self.depth_ranges)
                    star[3] = 0

                transform = self.transforms[star[3]]

                # Don't place a displayable if it's going to be invisible
                if transform.alpha <= 0.0:
                    continue

                # Convert the 3D coordinates to 2D using perspective projection.
                k = self.perspective / star[2]
                x = int(star[0] * k + self.origin_x)
                y = int(star[1] * k + self.origin_y)

                # Draw the star (if it's visible on screen).
                if 0 <= x < w and 0 <= y < h:
                    place(transform, x, y)

            renpy.redraw(self, 0)
            return render

    # See this: https://lemmasoft.renai.us/forums/viewtopic.php?f=8&t=37312&start=15
    class DisplayableSwitcher(renpy.Displayable, NoRollback):
        DEFAULT = {"d": Null(), "start_st": 0, "pause_st": 0, "force_pause": 0, "force_resume": 0}

        """
        This plainly switches displayable without re-showing the image/changing any variables by calling the change method.
        """

        def __init__(self, start_displayable="default", displayable=None, conditions=None, **kwargs):
            """
            Expects a dict of displayable={"string": something we can show in Ren'Py}

            Default is Null() unless specified otherwise.
            """
            super(DisplayableSwitcher, self).__init__(**kwargs)
            if not isinstance(displayable, dict):
                self.displayable = {"default": self.DEFAULT.copy()}
            else:
                self.displayable = {}
                for s, d in displayable.items():
                    self.displayable[s] = self.DEFAULT.copy()
                    d = renpy.easy.displayable(d)
                    if isinstance(d, ImageReference):
                        d = renpy.display.image.images[(d.name)]
                    self.displayable[s]["d"] = d
                    if isinstance(d, renpy.atl.ATLTransformBase):
                        self.displayable[s]["atl"] = d.copy()

                self.displayable["default"] = displayable.get("default", self.DEFAULT.copy())

            if not isinstance(conditions, (tuple, list)):
                self.conditions = None
            else:
                self.conditions = OrderedDict()
                for c, a in conditions:
                    code = renpy.python.py_compile(c, 'eval')
                    self.conditions[c] = a

            self.d = self.displayable[start_displayable]
            self.animation_mode = "normal"
            self.last_st = 0

        def per_interact(self):
            if self.conditions:
                for c, v in self.conditions.items():
                    if renpy.python.py_eval_bytecode(c):
                        s = v[0]
                        if len(v) > 1:
                            mode = v[1]
                        else:
                            mode = "normal"
                        self.change(s, mode)
                        break

        def change(self, s, mode="normal"):
            self.d = self.displayable[s]

            self.animation_mode = mode
            if mode == "reset":
                self.d["force_restart"] = 1
            elif mode == "pause":
                self.d["pause_st"] = self.last_st - self.d["start_st"]
            elif mode == "resume":
                self.d["force_resume"] = 1

        def render(self, width, height, st, at):
            if not st:
                for d in self.displayable.values():
                    d["start_st"] = 0
                    d["pause_st"] = 0

            rp = store.renpy

            self.last_st = st

            if self.animation_mode == "reset":
                if self.d["force_restart"]:
                    self.d["force_restart"] = 0
                    if "atl" in self.d:
                        self.d["d"].take_execution_state(self.d["atl"])
                        self.d["d"].atl_st_offset = st
                    else:
                        self.d["start_st"] = st
                st = st - self.d["start_st"] if not "atl" in self.d else st
            elif self.animation_mode in ("pause", "show_paused"):
                st = self.d["pause_st"]
            elif self.animation_mode == "resume":
                if self.d["force_resume"]:
                    self.d["force_resume"] = 0
                    self.d["start_st"] = st
                st = st - self.d["start_st"] + self.d["pause_st"]

            d = self.d["d"]
            cr = d.render(width, height, st, at)
            size = cr.get_size()
            render = rp.Render(size[0], size[1])

            try:
                position = d.get_placement()
                x, y = position[:2]
                if x is None:
                    x = 0
                if y is None:
                    y = 0
            except:
                x, y = 0, 0
            render.blit(cr, (x, y))

            rp.redraw(self, 0)
            return render

        def visit(self):
            return [v["d"] for v in self.displayable.values()]

    class LiveMarquee(renpy.display.layout.Container):

        def __init__(self, child, speed=200.0, style='tile', **properties):
            super(LiveMarquee, self).__init__(style=style, **properties)
            self.speed = speed
            self.add(child)

        def render(self, width, height, st, at):
            cr = renpy.display.render.render(self.child, width, height, st, at)
            cw = int(cr.get_size()[0])

            rv = renpy.display.render.Render(width, height)

            offset, _ = math.modf(st * self.speed / cw)
            x = -offset * cw
            while True:
                rv.blit(cr, (x, 0), focus=False)
                if x > width:
                    break
                x += cw

            renpy.redraw(self, 0)
            return rv

    def pull_warp(t):
        # This overshoots 1.0, so use only with transformations that can handle that.
        return 2*t - t**3

    def ease_warp_uncurried(t, T):
        return math.cos((T - t) * math.pi / 2.0)
    ease_warp = renpy.curry(ease_warp_uncurried)

    def ghost_disappear(trans, st, at):
        # Delay before disappear animation kicks in.
        st -= 1.2
        if st < 0.0:
            return -st

        zoom = 1.0 - 0.1 * st * (st + 1)
        alpha = 1.0 - 0.8 * st
        if zoom < 0.0 or alpha < 0.0:
            trans.alpha = 0.0
            return None
        else:
            trans.zoom = zoom
            trans.alpha = alpha
            return 0.0

    def animate_possess(ghost, host_name):
        # Determine position of the host and ghost sprites.
        host_x, host_y, host_w, host_h = renpy.get_image_bounds(host_name)

        scene_lists = renpy.scene_lists()
        ghost_displayable = scene_lists.get_displayable_by_tag('master', ghost.image_tag)

        dest = Position(
            xpos=absolute(host_x + 0.5 * host_w),
            ypos=absolute(host_y + getattr(renpy.store, host_name).ycenter * host_h),
            xanchor=0.5,
            yanchor=ghost.ycenter,
            )(sprite_of(ghost_displayable))
        dest = Transform(dest, function=ghost_disappear, xzoom=ghost.xzoom)
        return renpy.display.movetransition.MoveInterpolate(2.0, ghost_displayable, dest, False, time_warp=ease_warp(1.0))

    # Particle system used to visualize alien device activity.
    #
    # Usage:
    #     show expression alien_particles(<num>, <width>, <height>) as particles:
    #         xpos <xpos>
    #         ypos <xpos>
    #         alien_particles_fadeinout
    #     "<text>"
    #     hide particles

    alien_particle_raw = im.Image('images/alienparticle.png')
    alien_particle = im.MatrixColor(alien_particle_raw , im.matrix.tint(0.15, 0.3, 0.45))

    def alien_particles_update(sprites, scalefunc, st):
        xscale, yscale = (1.0, 1.0) if scalefunc is None else scalefunc(st)
        sin = math.sin
        modf = math.modf
        for sprite in sprites:
            sprite.x = xscale * sprite.xradius * sin(sprite.xphase + 3.0 * st)
            f, i = modf(sprite.yphase + 0.1 * st)
            sprite.y = yscale * sprite.yradius * ((1.0 - 2.0 * f) if i % 2 else (-1.0 + 2.0 * f))
        return 0.0

    def mind_scale(st, delay=1.0):
        st /= delay
        st -= 1.0
        if st < 0.0:
            f = (-st) ** 1.5
            return 1.0 + 2.0 * f, 1.0 + f
        else:
            return 1.0, 1.0

    def reject_scale(st):
        f = (2.0 * (st - 0.4)) ** 2
        return 1.0 + 2.0 * f, 1.0 + f

    def alien_particles_predict():
        return [alien_particle]

    def alien_particles(num, width, height, scalefunc=None):
        sprites = []
        manager = SpriteManager(
                update=renpy.partial(alien_particles_update, sprites, scalefunc),
                predict=alien_particles_predict,
                ignore_time=True)

        for i in range(num):
            sprite = manager.create(alien_particle)
            sprite.xphase = (float(i) / float(num)) * 2.0 * math.pi
            sprite.xradius = width * random.uniform(0.3, 0.7)
            sprite.yphase = random.uniform(0.0, 2.0 * math.pi)
            sprite.yradius = height * random.uniform(0.45, 0.55)
            sprites.append(sprite)

        particle_width, particle_height = renpy.image_size(alien_particle_raw)
        return Transform(manager,
                xoffset=-0.5 * particle_width,
                yoffset=-0.5 * particle_height,
                xanchor=0.0,
                yanchor=0.0,
                additive=1.0)

    # Particle system used to visualize meteor/ufo.
    #
    # Usage:
    #     show expression meteor_particles(<num>) as meteor:
    #         transform_anchor True
    #         zoom <zoom>
    #         pos (<xpos>, <ypos>)
    #         rotate <angle>

    tail_particle_raw = im.Image('images/tailparticle.png')
    tail_particle_blue = im.MatrixColor(tail_particle_raw , im.matrix.tint(0.02, 0.03, 0.08))
    tail_particle_purple = im.MatrixColor(tail_particle_raw , im.matrix.tint(0.04, 0.03, 0.08))
    tail_particle_size = (256, 16)
    tail_particle_blue = im.Scale(tail_particle_blue, *tail_particle_size)
    tail_particle_purple = im.Scale(tail_particle_purple, *tail_particle_size)
    tail_particle_lifetime = 8.0
    head_particle_blue = im.MatrixColor(tail_particle_raw , im.matrix.tint(0.25, 0.375, 1.0))
    head_particle_purple = im.MatrixColor(tail_particle_raw , im.matrix.tint(0.5, 0.375, 1.0))

    def tail_particle_reset(sprite):
        size = 16.0
        d = random.uniform(-1.0, 1.0)
        da = abs(d)
        y = size * d * da
        sprite.x = size - math.sqrt(size ** 2 - y ** 2)
        sprite.y = y
        r = random.uniform(0.1, 1.0) * (1.0 - 0.6 * da)
        sprite.speed = 2000.0 * (1.5 - r) ** 2
        sprite.maxage = r

    def tail_particles_update(sprites, st):
        aging = 1.0 / tail_particle_lifetime
        for sprite in sprites:
            t = (st - sprite.spawned) * aging
            if t > sprite.maxage:
                tail_particle_reset(sprite)
                sprite.spawned = st
            else:
                sprite.x = sprite.speed * t ** 1.5 + abs(sprite.y)
        return 0.0

    def tail_particles_predict(particle):
        return [particle]

    def tail_particles(num, tail_particle):
        sprites = []
        manager = SpriteManager(
                update=renpy.partial(tail_particles_update, sprites),
                predict=renpy.partial(tail_particles_predict, tail_particle),
                ignore_time=True)

        lifetime = tail_particle_lifetime
        for i in range(num):
            sprite = manager.create(tail_particle)
            tail_particle_reset(sprite)
            sprite.spawned = random.uniform(-lifetime * sprite.maxage, 0.0)
            sprites.append(sprite)

        return Transform(manager,
                yoffset=-tail_particle_size[1] / 2,
                subpixel=True,
                additive=1.0)

    def meteor_particles(num, head_particle, tail_particle):
        return Fixed(
            Transform(head_particle, xzoom=1.2, yzoom=1.5, yanchor=0.5, additive=1.0),
            tail_particles(num, tail_particle),
            xanchor=0.0, yanchor=0.0, rotate_pad=False)

    # Textures used to visualize magical activity.
    #
    # Usage:
    #     show pentagram at Position(pos=(placement_of(character).xpos, 0.5)) behind character as pentagram
    #     "<text>"
    #     hide pentagram with dissolve

    chalk_tint_purple = im.matrix.tint(0.5, 0.0, 0.3)
    chalk_tint_blue = im.matrix.tint(0.1, 0.0, 0.3)
    chalk_pentagram_purple = im.MatrixColor(im.Image('images/chalk-pentagram.png'), chalk_tint_purple)
    chalk_pentagram_blue = im.MatrixColor(im.Image('images/chalk-pentagram.png'), chalk_tint_blue)
    chalk_pentagram_circle_purple = im.MatrixColor(im.Image('images/chalk-pentagram-circle.png'), chalk_tint_purple)
    chalk_pentagram_circle_blue = im.MatrixColor(im.Image('images/chalk-pentagram-circle.png'), chalk_tint_blue)

    pentagram_ratio = (3 + math.sqrt(5)) / 2
    pentagram_size = 0.6
    pentagram_cycles = 4
    pentagram_speed = 2.0

    def pentagram_zoom(trans, st, at, cycle_offset, inner_image):
        phase, cycle = math.modf(st / pentagram_speed)
        cycle = int(cycle) + cycle_offset
        cycle_step = cycle % pentagram_cycles
        if cycle_step == 0:
            trans.alpha = 1.0 - (1.0 - phase) ** 2
        elif cycle_step == pentagram_cycles - 1:
            trans.alpha = 1.0 - phase ** 2
        else:
            trans.alpha = 1.0
        flip = cycle_offset % 2 == 0
        if pentagram_cycles % 2 == 1:
            flip ^= (cycle / pentagram_cycles) % 2
        zoom = pentagram_size * pentagram_ratio ** (cycle_step - 1 + phase)
        trans.xzoom = zoom
        trans.yzoom = zoom if flip else -zoom
        if trans.child is not inner_image and cycle >= pentagram_cycles:
            trans.set_child(inner_image)
        return 0

    def pentagram(outer_image, inner_image):
        def create(cycle_offset):
            return Transform(
                child=outer_image if cycle_offset == 0 else Null(),
                function=renpy.partial(pentagram_zoom, cycle_offset=cycle_offset, inner_image=inner_image),
                anchor=(0.5, 0.5),
                additive=1.0,
                subpixel=True)
        return Fixed(*(create(i) for i in range(pentagram_cycles)), anchor=(0.0, 0.0))

# Position transforms
transform left:
    xcenter 0.15 yalign 1.0

transform centerleft:
    xcenter 0.3 yalign 1.0

transform center:
    xcenter 0.5 yalign 1.0

transform centerright:
    xcenter 0.7 yalign 1.0

transform right:
    xcenter 0.85 yalign 1.0

# Flip transforms
transform faceleft:
    xzoom -1.0

transform faceright:
    xzoom 1.0

# Ghost transforms
transform anim_exspirit:
    alpha 0.0
    ease 0.5 alpha 1.0

# Other animations
transform pre_jump:
    ypos 1.04

transform jump_small:
    easein 0.12 ypos 1.0
    easeout 0.1 ypos 1.04

transform nudge_right(strength=30, speed=0.3):
    ease speed xoffset strength
    ease speed xoffset 0

transform nudge_left(strength=30, speed=0.3):
    ease speed xoffset -strength
    ease speed xoffset 0

transform standing:
    ypos 1.0

transform sitting:
    ypos 1.1

transform stand_up:
    sitting
    ease 0.5 standing

transform sit_down:
    standing
    ease 0.5 sitting

transform anim_fall_over_left:
    xzoom 1.0
    easeout 0.2 ypos 0.6 xanchor 0.5 yanchor 0.5
    parallel:
        easeout 0.7 rotate -90.0
    parallel:
        easeout 0.9 ypos 2.0

transform anim_fall_over_right:
    xzoom -1.0
    easeout 0.2 ypos 0.6 xanchor 0.5 yanchor 0.5
    parallel:
        easeout 0.7 rotate 90.0
    parallel:
        easeout 0.9 ypos 2.0

# Alien particle transforms
transform alien_particles_fadein:
    alpha 0.0
    linear 1.0 alpha 1.0

transform alien_particles_fadeinout(delay=1.0):
    alpha 0.0
    linear delay alpha 1.0
    pause delay
    linear delay alpha 0.0

transform alien_particles_fail:
    alpha 0.0
    linear 1.0 alpha 1.0
    linear 0.5 alpha 0.0

transform pentagram_rotate:
    transform_anchor True
    block:
        rotate 360
        linear 13.0 rotate 0
        repeat

image pentagram:
    chalk_pentagram_purple # for prediction only
    pentagram(chalk_pentagram_circle_purple, chalk_pentagram_purple)
    pentagram_rotate

# Variant that can be used to hint that a spell fails or is not working as intended.
image pentagram_oops:
    chalk_pentagram_blue # for prediction only
    pentagram(chalk_pentagram_circle_blue, chalk_pentagram_blue)
    pentagram_rotate

image pain:
    Solid('#ff0000')
    alpha 0.0
    linear 0.1 alpha 0.5
    linear 0.1 alpha 0.0

# Transforms for zooming into and out of faces for the scry function
transform scrychange_forward(new_widget, old_widget, x, y):
    delay 0.8
    contains:
        old_widget
        transform_anchor True
        xalign x
        yalign y
        ease 0.6 zoom 5.0
    contains:
        new_widget
        alpha 0.0
        pause 0.45
        ease 0.4 alpha 1.0

transform scrychange_backward(new_widget, old_widget, x, y):
    delay 0.6
    contains:
        old_widget
        alpha 1.0
        pause 0.6
        alpha 0.0
    contains:
        new_widget
        transform_anchor True
        xalign x
        yalign y
        zoom 5.0
        alpha 0.0
        parallel:
            ease 0.6 zoom 1.0
        parallel:
            ease 0.2 alpha 1.0

# Transforms for cross-fading bodies.
transform morph_sub_fadeout:
    linear 0.5 alpha 0.0

transform morph_tween_to_single(scale, y_pos=1.0):
    alpha 0.0
    ypos y_pos
    zoom scale
    parallel:
        easein 1.0 zoom 1.0
    parallel:
        easein 1.0 alpha 1.0

transform morph_tween_from_single(scale, y_pos=1.0):
    alpha 1.0
    ypos y_pos
    zoom 1.0
    parallel:
        easein 1.0 zoom scale
    parallel:
        easeout 1.0 alpha 0.0

transform morph_tween_to(scale):
    alpha 0.0
    zoom scale
    pause 0.5
    block:
        parallel:
            easein 1.0 zoom 1.0
        parallel:
            easein 1.0 alpha 1.0
        linear 0.5 alpha 0.0
        zoom scale
        repeat

transform morph_tween_from(scale):
    alpha 0.0
    zoom 1.0
    linear 0.5 alpha 1.0
    block:
        parallel:
            easein 1.0 zoom scale
        parallel:
            easeout 1.0 alpha 0.0
        zoom 1.0
        linear 0.5 alpha 1.0
        repeat

transform morph_pulse(time=0.5):
    alpha 0.0
    pause 0.0
    block:
        alpha 1.0
        zoom 1.0
        parallel:
            easeout time zoom 1.05
        parallel:
            easeout time alpha 0.0
        pause 1.0
        repeat

transform morph_pulse_rapid(y_pos=1.0):
    alpha 0.0
    ypos y_pos
    pause 1.0
    block:
        alpha 1.0
        zoom 1.0
        parallel:
            easeout 0.4 zoom 1.05
        parallel:
            easeout 0.4 alpha 0.0
        repeat

# Make entire screen flash white.
#
# Usage:
#     show white as flash:
#         additive_flash(<delay>)
#     "<text>"
#     hide flash

transform additive_flash(delay):
    additive 1.0
    alpha 0.0
    easein delay alpha 0.5
    easeout delay alpha 0.0

transform flare_up:
    alpha 0.0
    xzoom 0.0
    yzoom 0.0
    parallel:
        linear 0.2 yzoom 2.0
        pause 0.15
        linear 0.2 yzoom 0.0
    parallel:
        linear 0.15 xzoom 2.0
        pause 0.15
        linear 0.3 xzoom 0.0
    parallel:
        ease 0.3 alpha 1.0
    parallel:
        ease 0.4 additive 0.8

transform zap_out:
    xanchor 0.5
    yanchor 0.5
    ypos 0.5
    parallel:
        linear 0.08 yzoom 1.4
    parallel:
        linear 0.08 xzoom 0.5
    parallel:
        pause 0.08
        ease 0.14 yzoom 0.0
    parallel:
        pause 0.1
        linear 0.1 xzoom 1.0
        linear 0.1 xzoom 0.0
    parallel:
        pause 0.1
        linear 0.3 alpha 0.0
    parallel:
        linear 0.2 additive 1.0

init python:
    # Moves a displayable to a position from a small offset while fading it in.
    # Unfortunately the "function" statement can not pass parameters, so we
    # have to employ a trick to access them, as can be seen below.
    # This function is necessary because we want to handle multiple types
    # for the "pos" parameter, which normal transforms are not built to handle.
    def _fade_in_side(trans, st, at):
        pos = trans.context.context["pos"] # Final position of the displayable
        speed = trans.context.context["speed"] # Duration of the animation
        left = trans.context.context["left"] # Whether to slide in from the left or right side
        # Access the "xcenter" property of the passed transform (if it is one)
        # This is position-dependant! If you create a new transform to use with this,
        # make sure that xcenter comes before all other properties!
        value = pos.properties[0][1] if isinstance(pos, renpy.display.motion.ATLTransform) else pos
        perc = st / speed # make animation easier by converting to percent
        trans.yalign = 1.0
        trans.alpha = perc * 1.0 # animate alpha
        offset = 0.2 * (-1 if left else 1)
        trans.xcenter = (value + offset) - (perc * offset) # animate position
        if perc >= 1: # If animation finished: Stop and never call us again
            trans.alpha = 1.0
            trans.xcenter = value
            return None
        return 0 # Call us as soon as possible (time in seconds)

    # yes, this is a copy of the above function.
    # this is not one function because it makes it easier to adjust fadein and fadeout animations
    # separately, which might be required in the future.
    def _fade_out_side(trans, st, at):
        pos = trans.context.context["pos"] # Final position of the displayable
        speed = trans.context.context["speed"] # Duration of the animation
        left = trans.context.context["left"] # Whether to slide in from the left or right side
        # Access the "xcenter" property of the passed transform (if it is one)
        # This is position-dependant! If you create a new transform to use with this,
        # make sure that xcenter comes before all other properties!
        value = pos.properties[0][1] if isinstance(pos, renpy.display.motion.ATLTransform) else pos
        perc = st / speed # make animation easier by converting to percent
        trans.yalign = 1.0
        trans.alpha = 1 - (perc * 1.0) # animate alpha
        offset = 0.2 * (-1 if left else 1)
        trans.xcenter = (value + offset) - ((1 - perc) * offset) # animate position
        if perc >= 1: # If animation finished: Stop and never call us again
            trans.alpha = 0.0
            trans.xcenter = value
            return None
        return 0 # Call us as soon as possible (time in seconds)

# "pos" can be either a float (0.0 to 1.0)
# or one of the ATLTransforms we defined in script.rpy:
# - left, centerleft, center, centerright, right
transform fadeinright(pos, speed=0.2, left=False):
    function _fade_in_side
    alpha 1.0

transform fadeinleft(pos, speed=0.27, left=True):
    function _fade_in_side
    alpha 1.0

# remember that after the fadeout, the sprite is still onscreen,
# it is simply transparent. you should always "hide" it after using
# this transition.
transform fadeoutright(pos, speed=0.2, left=False):
    function _fade_out_side
    alpha 0.0

transform fadeoutleft(pos, speed=0.27, left=True):
    function _fade_out_side
    alpha 0.0

transform phone_call_icon_dialing:
    xalign 0.5
    yalign 0.5
    zoom 0.0
    alpha 0.0
    easein 0.29 zoom 1.2 alpha 1.0
    easein 0.18 zoom 0.9
    easein 0.25 zoom 1.0

transform phone_call_icon_picking_up:
    xalign 0.5
    yalign 0.5
    zoom 1.0
    easein 0.4 zoom 1.3
    easein 0.2 zoom 0.9
    easein 0.3 zoom 1.0
    ease 0.8 rotate 360

transform phone_call_separator_in_top:
    xalign 0.5
    ypos -720
    ease 0.6 ypos -360

transform phone_call_separator_in_bottom:
    xalign 0.5
    ypos 720
    ease 0.6 ypos 360

# Transitions:
init python:
    def HorizWipeTransition(time=0.5, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/wipe-horiz.png'), time, reverse=reverse)

    def VertWipeTransition(time=0.5, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/wipe-vert.png'), time, reverse=reverse)

    def HorizSawtoothTransition(time=0.5, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/sawtooth-horiz.png'), time, ramplen=64, reverse=reverse)

    def CircleTransition(time=1.0, reverse=False):
        return ImageDissolve('images/transitions/circle.png', time, reverse=reverse)

    def HorizBlindsTransition(time=0.5, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/blinds-horiz.png'), time, reverse=reverse)

    def VertBlindsTransition(time=0.5, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/blinds-vert.png'), time, reverse=reverse)

    def HorizCenterTransition(time=0.5, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/center-horiz.png'), time, reverse=reverse)

    def VertCenterTransition(time=1.0, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/center-vert.png'), time, reverse=reverse)

    def ClockTransition(time=1.0, reverse=False, flip=False):
        if flip:
            img = im.Flip('images/transitions/clock.png', vertical=True)
        else:
            img = 'images/transitions/clock.png'
        return ImageDissolve(img, time, reverse=reverse)

    def DiamondsTransition(time=0.5, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/diamonds.png'), time, reverse=reverse)

    def HorizRadialTransition(time=1.0, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/radial-horiz.png'), time, reverse=reverse)

    def VertRadialTransition(time=1.0, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/radial-vert.png'), time, reverse=reverse)

    def CubeCloudsTransition(time=1.0, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/cubeclouds.png'), time, reverse=reverse)

    def PhaseTransition(time=1.0, reverse=False):
        return ImageDissolve(im.Tile('images/transitions/phase.png'), time, reverse=reverse)

    def DissolveMove(delay, layers, enter=None, leave=None, time_warp=None,
                    enter_time_warp=None, leave_time_warp=None):
        top = Dissolve(delay, time_warp=lambda x: min(x / (0.18 / delay), 1))
        before = MoveTransition(delay, layers=layers, enter=enter, leave=leave, time_warp=time_warp, old=True,
                                enter_time_warp=enter_time_warp, leave_time_warp=leave_time_warp)
        after = MoveTransition(delay, layers=layers, enter=enter, leave=leave, time_warp=time_warp,
                                enter_time_warp=enter_time_warp, leave_time_warp=leave_time_warp)
        return ComposeTransition(top, before=before, after=after)

    def dissolve_move_transition(prefix, delay, time_warp=None, in_time_warp=None, out_time_warp=None, layers=["master"]):
        moves = {
            "" : DissolveMove(delay, layers=layers, time_warp=time_warp),
            "inright" : DissolveMove(delay, layers=layers, enter=_moveright,
                                    time_warp=time_warp, enter_time_warp=in_time_warp),
            "inleft" : DissolveMove(delay, layers=layers, enter=_moveleft,
                                    time_warp=time_warp, enter_time_warp=in_time_warp),
            "intop" : DissolveMove(delay, layers=layers, enter=_movetop,
                                    time_warp=time_warp, enter_time_warp=in_time_warp),
            "inbottom" : DissolveMove(delay, layers=layers, enter=_movebottom,
                                    time_warp=time_warp, enter_time_warp=in_time_warp),
            "outright" : DissolveMove(delay, layers=layers, leave=_moveright,
                                    time_warp=time_warp, leave_time_warp=out_time_warp),
            "outleft" : DissolveMove(delay, layers=layers, leave=_moveleft,
                                    time_warp=time_warp, leave_time_warp=out_time_warp),
            "outtop" : DissolveMove(delay, layers=layers, leave=_movetop,
                                    time_warp=time_warp, leave_time_warp=out_time_warp),
            "outbottom" : DissolveMove(delay, layers=layers, leave=_movebottom,
                                    time_warp=time_warp, leave_time_warp=out_time_warp),
        }

        for k, v in moves.items():
            setattr(store, prefix + k, v)

    dissolve_move_transition("ease", 0.5, _ease_time_warp, _ease_in_time_warp, _ease_out_time_warp)

    # Disable lint warnings about replacing wipes
    config.lint_ignore_replaces.extend(('wipeup', 'wipedown', 'wipeleft', 'wiperight'))

    # Monkey-patch the ability to insert custom durations for standard transforms
    def delay_insert(move, delay=0.5):
        moves = {
            "move" : MoveTransition(delay),
            "moveinright" : MoveTransition(delay, enter=_moveright),
            "moveinleft" : MoveTransition(delay, enter=_moveleft),
            "moveintop" : MoveTransition(delay, enter=_movetop),
            "moveinbottom" : MoveTransition(delay, enter=_movebottom),
            "moveoutright" : MoveTransition(delay, leave=_moveright),
            "moveoutleft" : MoveTransition(delay, leave=_moveleft),
            "moveouttop" : MoveTransition(delay, leave=_movetop),
            "moveoutbottom" : MoveTransition(delay, leave=_movebottom),
            "ease" : MoveTransition(delay, time_warp=_ease_time_warp, enter_time_warp=_ease_in_time_warp),
            "easeinright" : MoveTransition(delay, enter=_moveright, time_warp=_ease_time_warp, enter_time_warp=_ease_in_time_warp),
            "easeinleft" : MoveTransition(delay, enter=_moveleft, time_warp=_ease_time_warp, enter_time_warp=_ease_in_time_warp),
            "easeintop" : MoveTransition(delay, enter=_movetop, time_warp=_ease_time_warp, enter_time_warp=_ease_in_time_warp),
            "easeinbottom" : MoveTransition(delay, enter=_movebottom, time_warp=_ease_time_warp, enter_time_warp=_ease_in_time_warp),
            "easeoutright" : MoveTransition(delay, leave=_moveright, time_warp=_ease_time_warp, leave_time_warp=_ease_out_time_warp),
            "easeoutleft" : MoveTransition(delay, leave=_moveleft, time_warp=_ease_time_warp, leave_time_warp=_ease_out_time_warp),
            "easeouttop" : MoveTransition(delay, leave=_movetop, time_warp=_ease_time_warp, leave_time_warp=_ease_out_time_warp),
            "easeoutbottom" : MoveTransition(delay, leave=_movebottom, time_warp=_ease_time_warp, leave_time_warp=_ease_out_time_warp),
        }
        return moves[move]

    st_move = renpy.partial(delay_insert, "move")
    st_moveinbottom = renpy.partial(delay_insert, "moveinbottom")
    st_moveinleft = renpy.partial(delay_insert, "moveinleft")
    st_moveinright = renpy.partial(delay_insert, "moveinright")
    st_moveintop = renpy.partial(delay_insert, "moveintop")
    st_moveoutbottom = renpy.partial(delay_insert, "moveoutbottom")
    st_moveoutleft = renpy.partial(delay_insert, "moveoutleft")
    st_moveoutright = renpy.partial(delay_insert, "moveoutright")
    st_moveouttop = renpy.partial(delay_insert, "moveouttop")

    st_ease = renpy.partial(delay_insert, "ease")
    st_easeinbottom = renpy.partial(delay_insert, "easeinbottom")
    st_easeinleft = renpy.partial(delay_insert, "easeinleft")
    st_easeinright = renpy.partial(delay_insert, "easeinright")
    st_easeintop = renpy.partial(delay_insert, "easeintop")
    st_easeoutbottom = renpy.partial(delay_insert, "easeoutbottom")
    st_easeoutleft = renpy.partial(delay_insert, "easeoutleft")
    st_easeoutright = renpy.partial(delay_insert, "easeoutright")
    st_easeouttop = renpy.partial(delay_insert, "easeouttop")

define wipeleft = HorizWipeTransition()
define wiperight = HorizWipeTransition(reverse=True)
define wipedown = VertWipeTransition()
define wipeup = VertWipeTransition(reverse=True)
define sawleft = HorizSawtoothTransition()
define sawright = HorizSawtoothTransition(reverse=True)
define wipecircle = CircleTransition()
define wipeblinds_horiz = HorizBlindsTransition()
define wipeblinds_vert = VertBlindsTransition()
define wipecenter_horiz = HorizCenterTransition()
define wipecenter_vert = VertCenterTransition()
define wipeclock = ClockTransition()
define wiperadial_horiz = HorizRadialTransition()
define wiperadial_vert = VertRadialTransition()
define diamonds = DiamondsTransition()
define cubeclouds = CubeCloudsTransition()
define phase = PhaseTransition()
define long_hpunch = Move((15, 0), (-15, 0), .10, bounce=True, repeat=True, delay=0.55)
define hpunch_small = Move((5, 0), (-5, 0), .10, bounce=True, repeat=True, delay=0.275)
init +10 python:
    if persistent.not_with_exchange:
        renpy.store.exchange = None
    else:
        renpy.store.exchange = {"master": Dissolve(0.18)}
define fdissolve = Dissolve(0.18)

# Static backgrounds
image black = Solid((0, 0, 0, 255))
image white = Solid((255, 255, 255, 255))
image blur = Solid((255, 255, 255, 200))

# Effects images
image flare = "images/flare.png"
image logo = "gui/arrows.png"

# Animated Effects
image line_action_out:
    "images/line_action/out/1.png"
    pause 0.1
    "images/line_action/out/2.png"
    pause 0.1
    repeat

image line_action_in:
    "images/line_action/in/1.png"
    pause 0.1
    "images/line_action/in/2.png"
    pause 0.1
    repeat

# Phone system
image phone modern = "gui/messaging_system/phone_modern.png"
image phone_tip_pink = "gui/messaging_system/bubble_tip_pink.png"
image phone_tip_green = "gui/messaging_system/bubble_tip_green.png"

# Intro animation backgrounds
image intro_trees = "gui/intro/trees.png"
image intro_panorama = "gui/intro/panorama.png"
image intro_silhouettes = "gui/intro/silhouettes.png"
