# !/usr/bin/env python

from manimlib.imports import *
import numpy as np

class definitions(Scene):
    def construct(self):
        mobius_title = TextMobject(r"The ", r"M\"obius function", r" is given as,")
        mobius_title.set_color_by_tex(r"M\"obius function", BLUE)
        mobius_title.to_edge(UP)

        mobius = TexMobject(
            r"\mu(n)",  "=", r""" \begin{cases}
                        1, \quad n = 1, \\
                        (-1)^k, \quad n = \prod_{i = 1}^k p_i \text{ distinct primes}  \\
                        0, \quad n \text{ not squarefree}
            \end{cases} """
            )

        mobius.set_color_by_tex(r"\mu", BLUE)
        mobius.next_to(mobius_title, direction=DOWN)

        mertens_title = TextMobject("The ", r"""Mertens function """, "is given as,")
        mertens_title.set_color_by_tex(r"function", GREEN)
        mertens_title.move_to(mobius_title.get_center())

        mertens_list = ["M(n)", "=", "\sum_{k=1}^n", "\mu(k)", r", \quad", "n", "\in", "\Z^+"]
        mertens = TexMobject(*mertens_list)
        
        mertens.set_color_by_tex("M(n)", GREEN)
        mertens.set_color_by_tex(r"\mu", BLUE)
        mertens.next_to(mertens_title, direction=DOWN)
        mertens.shift(RIGHT)

        primes = "distinct primes"
        loc = str(mobius[2]).find(primes)

        self.play(
            Write(mobius_title),
            FadeInFrom(mobius, UP))
        self.wait()
        self.play(Indicate(mobius[2][27 + loc:25 + len(primes)]))
        self.wait()

        mobius.generate_target()
        mobius.target.scale(.8)
        mobius.target.to_corner(DR)
        self.play(MoveToTarget(mobius))
        self.play(FadeOutAndShiftDown(mobius_title))

        self.play(
            Write(mertens_title),
            Write(mertens))
        self.wait(2)

        mertens.generate_target()
        mertens.target.to_corner(UL)


        self.play(
            FadeOutAndShiftDown(mertens_title),
            MoveToTarget(mertens)
        )
        self.wait(2)

        cts_mert_list = ["M(x)", "=", r"M( \lfloor x \rfloor)", r",\quad x \in \R^+"]

        cts_mert = TexMobject(*cts_mert_list)
        cts_mert.next_to(mertens, direction=DOWN)
        cts_mert.to_edge(LEFT)
        cts_mert.set_color_by_tex("M(x)", GREEN)
        cts_mert.set_color_by_tex(r"M( \lfloor x \rfloor)", GREEN)

        self.play(Transform(copy.deepcopy(mertens), cts_mert))

        conjec = TexMobject(r"|M(x)|", "<", "\sqrt{x}")
        conjec.set_color_by_tex(r"|M(x)|", GREEN)
        conjec.next_to(cts_mert, direction=DOWN, buff=.5)
        conjec.to_edge(LEFT)

        self.wait(2)
        
        self.play(Write(conjec))

        self.wait(2)

        title = TextMobject("Mertens Conjecture")
        title.next_to(conjec, buff=1)
        title.set_color(YELLOW)
        self.play(Write(title))

class RH(Scene):
    def construct(self):
        zeta = TexMobject(r"\frac{1}{\zeta(s)}", "=", r"\sum^{\infty} {" , r"\mu(n)", r"\over n^s }")
        zeta.set_color_by_tex(r"\mu", BLUE)
        self.play(Write(zeta))

        zeta.generate_target()
        
        mert_integral = TexMobject(r"M(x)", r"= \frac{1}{2 \pi i} \int_{\sigma - i \infty}^{\sigma + i \infty} \frac{x^s}{s \zeta(s)}")
        mert_integral[0].set_color(GREEN)
        implies = TexMobject(r"\Rightarrow")
        
        implies.next_to(zeta.target, buff=1)
        mert_integral.next_to(implies, buff=1)

        group = VGroup(zeta.target, implies, mert_integral)
        group.to_edge(UP)
        group.set_x(0)

        self.play(MoveToTarget(zeta), LaggedStart(Write(implies)), LaggedStart(Write(mert_integral)))
        self.wait(2)

        conj = TextMobject("Mertens Conjecture")
        conj.set_color(YELLOW)
        rh = TextMobject("Riemann Hypothesis")
        down_arrow = TexMobject(r"\Downarrow")
        conj.next_to(group, direction=DOWN)
        down_arrow.next_to(conj, direction=DOWN)
        rh.next_to(down_arrow, direction=DOWN)

        self.play(*[Write(a) for a in [rh, down_arrow, conj]])
        self.wait(2)

        conj_and_rh = VGroup(conj, down_arrow, rh)
        conj_and_rh.generate_target()
        conj_and_rh.target.to_edge(UP)

        self.play(MoveToTarget(conj_and_rh),FadeOutAndShift(group), FadeOutAndShift(zeta))
        self.wait(2)

        disproof = TextMobject(r"Odlyzko and te Riele (1985) disprove the ", "Mertens Conjecture", r" \\ using LLL (1982) applied to Diophantine approximation")
        disproof.next_to(conj_and_rh, direction=DOWN)
        disproof.set_color_by_tex("Mertens", YELLOW)
        self.play(Write(disproof))
        self.wait(2)

        liminf = TexMobject(r"\liminf {", r"M(n)", r"\over \sqrt{n} } < -1.009")
        limsup = TexMobject(r"\limsup {", r"M(n)", r"\over \sqrt{n} } > 1.06")

        liminf[1].set_color(GREEN)
        limsup[1].set_color(GREEN)

        limsup.next_to(disproof, direction=DOWN)
        liminf.next_to(limsup, direction=DOWN)

        self.play(*[Write(o) for o in [limsup, liminf]])
        self.wait(2)

        counter = TexMobject(r"\text{First counterexample } \in \left( 10^{16}, e^{1.59 \times 10^{40}} \right)")
        counter.to_edge(DOWN)
        self.play(Write(counter))
        self.wait(5)

        self.play(*[FadeOut(o) for o in [disproof, limsup, liminf, counter]])

        like = TextMobject("Mertens-like Conjecture")
        like.set_color(YELLOW)
        like.move_to(conj.get_center())

        self.play(Transform(conj, like))
        self.wait(2)

        lambda_mert = TexMobject(r"\left| M(x) \right|", r" < \lambda \sqrt{x}")
        lambda_mert.next_to(conj, direction=DOWN)
        lambda_mert[0].set_color(GREEN)

        arrow_and_rh = VGroup(down_arrow, rh)
        arrow_and_rh.generate_target()
        arrow_and_rh.target.next_to(lambda_mert, direction=DOWN)

        self.play(MoveToTarget(arrow_and_rh), Write(lambda_mert))
        self.wait(2)
        
        more_compute = TextMobject(r"Odlyzko and te Riele claim that \\ more computation could disprove these")
        more_compute.next_to(rh, direction=DOWN)
        self.play(Write(more_compute))
        self.wait(2)

class GraphMert(GraphScene):
    CONFIG={
        "x_min": 0,
        "x_max": 10000000,
        "x_axis_label": "$x$",
        "y_axis_label": "$y$",
        "function_color": RED,
        "graph_origin": UP*.4 + LEFT*3.5,
    }

    CONFIG["y_max"] = int(np.sqrt(CONFIG["x_max"]))
    CONFIG["y_min"] = -1 * CONFIG["y_max"]
    CONFIG["x_labeled_nums"] = range(0, int(CONFIG["x_max"] + 1), int(CONFIG["x_max"] / 2))
    CONFIG["y_labeled_nums"] = range(int(CONFIG["y_min"]), int(CONFIG["y_max"]) + 1, int(CONFIG["y_max"] / 2))
    CONFIG["x_tick_frequency"] = int(CONFIG["x_max"] / 4)
    CONFIG["y_tick_frequency"] = int(CONFIG["y_max"] / 2)

    def primes(self, n):
        prime_fact = {}
        d = 2
        while d*d <= n:
            while (n % d) == 0:
                if d not in prime_fact:
                    prime_fact[d] = 1
                else:
                    prime_fact[d] += 1 
                n //= d
            d += 1
        if n > 1:
            prime_fact[n] = 1
        return prime_fact

    def mobius(self, n):
        if n == 1:
            return 1
        fact = self.primes(n)

        for power in fact.values():
            if power > 1:
                return 0
        
        number_of_facts = len(fact.keys())

        if number_of_facts % 2 == 0:
            return 1 
        else: 
            return -1

    def mertens_function(self, n):
        n = np.floor(n)
        if n in self.memo.keys():
            return self.memo[n]

        if n <= 0:
            self.memo[0] = 0
            return 0
        if n == 1:
            self.memo[1] = 1
            return 1
        if n - 1 in self.memo.keys():
            self.memo[n] = self.memo[n - 1] + self.mobius(n)
            return self.memo[n]

    def top_sqrt(self, x):
        return np.sqrt(x)

    def bot_sqrt(self, x):
        return -np.sqrt(x)

    def construct(self):
        self.memo = {}
        coords = [(x, self.mertens_function(x)) for x in range(0, self.x_max)]
        
        self.setup_axes(animate=True)

        func_graph = self.get_graph(self.mertens_function,
                                    self.function_color)
        func_graph_2 = self.get_graph(self.top_sqrt, GREEN)
        func_graph_3 = self.get_graph(self.bot_sqrt, GREEN)
        
        self.play(ShowCreation(func_graph))
        self.wait(2)
        self.play(ShowCreation(func_graph_2), ShowCreation(func_graph_3))
    