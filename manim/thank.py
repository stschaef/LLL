from manimlib.imports import *

class Ending(Scene):
    def construct(self):
        a = TextMobject("Thank you for watching!")

        self.play(Write(a))
        self.wait(1)
        
        me = TextMobject("Made by Steven Schaefer")
        me.next_to(a, direction=DOWN)
        self.play(Write(me))
        self.wait(1)