from manimlib.imports import *
import numpy as np

class Key(SVGMobject):
    def __init__(self):
        SVGMobject.__init__(self, file_name="key.svg")
        self.scale(.37)

class PublicKey(GraphScene):
    def construct(self):
        snoopy = SVGMobject("snoopy.svg")
        snoopy.set_color(BLUE_C)
        snoopy.scale(1.5)
        snoopy.to_corner(UL)
        snoopy.shift(LEFT*.2)

        woodstock = SVGMobject("Woodstock.svg")
        woodstock.to_corner(UR)
        woodstock.set_color(YELLOW_C)
        woodstock.scale(1.5)
        woodstock.shift(RIGHT*2)
        woodstock.shift(DOWN*.25)
        self.play(Write(snoopy), Write(woodstock))
        self.wait(1)

        ntru = TexMobject(r"\textbf{NTRU}")
        ntru.scale(2)

        self.play(Write(ntru))
        self.wait(2)
        self.play(FadeOut(ntru))

        public = TextMobject(r"\underline{Public}")
        self.play(Write(public))
        self.wait(2)

        params = TexMobject("(", "N", ",", r"p", ",", r"q", ")")
        params.next_to(public, direction=DOWN)

        conditions = TexMobject("N", r"\text{ prime, }", "p", "<", "q", r",\ ", r"\text{gcd}(", "p", ",","q", ") = 1")
    
        conditions.to_edge(UP)
        self.play(Write(params))
        self.play(Write(conditions))
        self.wait(2)
        self.play(FadeOutAndShift(conditions))

        private = TextMobject(r"\underline{Private}")
        private.to_edge(RIGHT)
        self.play(Write(private))

        polys = TexMobject("f", ",", "g")
        polys.set_color_by_tex("f", BLUE)
        polys.set_color_by_tex("g", PINK)
        poly_conditions = TexMobject("f", ",", "g", r"\in R[X]/(X^N - 1)")
        poly_conditions.set_color_by_tex("f", BLUE)
        poly_conditions.set_color_by_tex("g", PINK)
        poly_conditions.set_color_by_tex(r"\text", WHITE)

        polys.next_to(private, direction=DOWN)
        poly_conditions.to_edge(UP)

        coeffs_cond = TexMobject(r"\text{ random, coeffiecients in }", r"\{-1,0,1\}")
        coeffs_cond.next_to(poly_conditions, direction=DOWN)
        coeffs_cond.shift(RIGHT*.15)

        invertible = TexMobject("f", r"\text{ invertible }", r"\text{mod } p, \text{mod } q")
        invertible.set_color_by_tex("f", BLUE)
        invertible.next_to(coeffs_cond, direction=DOWN)


        self.play(Write(polys))
        self.wait(2)
        self.play(Write(poly_conditions), Write(coeffs_cond), Write(invertible))
        self.wait(3)

        private_key = Key()
        private_key.set_color(YELLOW)
        private_key.next_to(polys, direction=LEFT)
        self.play(Write(private_key))

        self.play(*[FadeOutAndShift(o) for o in [poly_conditions, coeffs_cond, invertible]])
        self.wait(2)

        h = TexMobject("h", r"\equiv", r"p \cdot", r"f_p", "g", r" \pmod  q")
        h.set_color_by_tex("f", BLUE)
        h.set_color_by_tex("g", PINK)
        h.set_color_by_tex("h", ORANGE)

        h.to_edge(UP)
        self.play(Write(h))
        self.wait(2)

        h_pub = TexMobject(r"\text{Public key }", "h")
        h_pub[1].set_color(ORANGE)
        h_pub.next_to(params, direction=DOWN)

        pub_key = Key()
        pub_key.set_color(ORANGE)
        pub_key.next_to(h_pub, direction=LEFT)

        self.play(Transform(h, h_pub), Write(pub_key))
        self.wait(2)

        msg = TexMobject("m", r"\text{ message as a polynomial}")
        msg[0].set_color(GREEN)
        msg_conds = TextMobject("coefficients in $[-p/2, p/2]$")
        rand_r = TexMobject(r"\text{random polynomial", "r")

        msg.to_edge(UP)
        msg_conds.next_to(msg, direction=DOWN)
        rand_r.next_to(msg_conds, direction=DOWN)

        private_snoop = TextMobject(r"\underline{Private}")
        private_snoop.to_edge(LEFT)
        self.play(Write(private_snoop))

        msg_snoop = TexMobject(r"\text{Message }", "m")
        msg_snoop[1].set_color(GREEN)
        msg_snoop.next_to(private_snoop, direction=DOWN)
        msg_snoop.to_edge(LEFT)

        r_snoop = TexMobject(r"\text{Random }", "r")
        r_snoop.next_to(msg_snoop, direction=DOWN)
        r_snoop.to_edge(LEFT)

        self.play(*[Write(o) for o in [msg, msg_conds, rand_r]])
        self.wait(2)

        self.play(FadeOutAndShift(msg_conds), Transform(rand_r, r_snoop), Transform(msg, msg_snoop))

        e = TexMobject("e", r"\equiv", "r", "h", "+", "m", r" \pmod q")
        e[0].set_color(GREEN)
        e[5].set_color(GREEN)
        e[3].set_color(ORANGE)

        e.to_edge(UP)
        self.play(Write(e))
        self.wait(2)

        e_snoop = TexMobject(r"\text{Encrypted }", "e")
        e_snoop[1].set_color(GREEN)
        e_snoop.next_to(rand_r, direction=DOWN)
        e_snoop.to_edge(LEFT)

        self.play(Transform(e, e_snoop))
        self.wait(2)

        msg_e = TexMobject("e")
        msg_e.set_color(GREEN)
        msg_e.scale(2)
        msg_e.next_to(snoopy, buff=1)
        orig_y = msg_e.get_y()


        mail = SVGMobject("mail.svg")
        mail.scale(.5)
        mail.set_color(GREEN)
        mail.next_to(msg_e, direction=DOWN)

        mail.add_updater(lambda obj: obj.next_to(msg_e, direction=DOWN))

        
        self.play(Write(mail), ReplacementTransform(copy.deepcopy(e_snoop), msg_e))
        self.wait(2)

        msg_e.generate_target()
        msg_e.target.next_to(woodstock, direction=LEFT, buff=1)
        msg_e.target.set_y(orig_y)

        self.play(MoveToTarget(msg_e))

        self.wait(2)

        a_eq = TexMobject(r"a \equiv ", "f", "e", r" \pmod q")
        a_eq[1].set_color(BLUE)
        a_eq[2].set_color(GREEN) 
        a_eq.to_edge(UP)

        b_eq = TexMobject(r"b \equiv a \pmod p \equiv f \cdot ", "m", r" \pmod p")
        b_eq[1].set_color(GREEN)
        b_eq.next_to(a_eq, direction=DOWN)
        b_eq.shift(RIGHT*.22)


        c_eq = TexMobject(r"c \equiv ", "f_p", r" \cdot b \equiv ", "m", r" \pmod p")
        c_eq[1].set_color(BLUE)
        c_eq[3].set_color(GREEN)
        c_eq.next_to(b_eq, direction=DOWN)

        self.play(FadeOut(msg_e), FadeOut(mail))
        self.play(Write(a_eq))
        self.wait(2)
        self.play(Write(b_eq))
        self.wait(2)
        self.play(Write(c_eq))
        self.wait(2)
        
        big_m = TexMobject("m")
        big_m.scale(3.5)
        big_m.set_color(GREEN)
        
        group = VGroup(a_eq, b_eq, c_eq)
        big_m.move_to(group.get_center())
        self.play(Transform(group, big_m))
        self.wait(2)

        lattice_stuff = TextMobject("Operations in a lattice!")
        lattice_stuff.scale(2)
        lattice_stuff.to_edge(DOWN)
        self.play(Write(lattice_stuff))
        self.wait(2)

        
        