#!/usr/bin/env python

from manimlib.imports import *

# To watch one of these scenes, run the following:
# python -m manim example_scenes.py SquareToCircle -pl
#
# Use the flag -l for a faster rendering at a lower
# quality.
# Use -s to skip to the end and just save the final frame
# Use the -p to have the animation (or image, if -s was
# used) pop up once done.
# Use -n <number> to skip ahead to the n'th animation of a scene.
# Use -r <number> to specify a resolution (for example, -r 1080
# for a 1920x1080 video)

class onering(Scene):
    def construct(self):

        no = TextMobject("NO!", color=RED)

        text = TextMobject(
            "$\R$",     #0
            " is a ",   #1
            "ring?"
        )

        module = TextMobject(
            "$\R$",     #0
            " is a ",   #1
            "$\Q$-module!"
        )

        only = TextMobject("$\Z$ is the only ring!", color=YELLOW)


        no.scale(3.5)
        text.scale(3.5)
        module.scale(3.5)
        only.scale(3.5)

        no.next_to(text, DOWN)

        self.play(Write(text))
        self.wait()
        self.play(Write(no))
        self.wait()
        self.play(FadeOut(no))
        self.wait()
        self.play(Transform(text, module))
        self.wait(2)
        self.play(Transform(text, only))
        self.wait(2)

class vectors(LinearTransformationScene):
    def construct(self):
        grid = NumberPlane()
        grid_title = TextMobject("This is a grid")
        grid_title.scale(1.5)
        grid_title.to_corner(UP + LEFT)
        
        self.add(grid, grid_title)  # Make sure title is on top of grid
        self.play(
            # FadeInFromDown(grid_title)
            ShowCreation(grid, run_time=3, lag_ratio=0.1)
        )
        self.wait()
        self.add_vector(np.array([1,1]))