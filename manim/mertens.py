# !/usr/bin/env python

from manimlib.imports import *
import numpy as np

class definitions(Scene):
    def construct(self):
        mobius_title = TextMobject("The ", r"""M\"obius function """, "is given as,")
        mobius_title[1].set_color(BLUE)
        mobius_title.shift(UP * 3.5)

        mobius = TexMobject(
            "\mu", "(", "n", ")",  "=", r""" \begin{cases}
                        1, \quad n = 1, \\
                        (-1)^k, \quad n = \prod_{i = 1}^k p_i \text{ distinct primes}  \\
                        0, \quad n \text{ not squarefree}
            \end{cases} """
            )
        for i in [0, 1, 3]:
            mobius[i].set_color(BLUE)
        mobius[2].set_color(RED)
        for i, char in enumerate(mobius[5]):
            if char.path_string == TexMobject("n")[0][0].path_string:
                mobius[5][i].set_color(RED)
            
        mobius.shift(UP)

        mertens = TexMobject(
            "M", "(", "n", ")", "=", "\sum_{k=1}^n", "\mu(k)", r", \quad", "n", "\in", "\Z^+"
        )
        
        for i in [0, 1, 3]:
            mertens[i].set_color(GREEN)
        for i in [2, 8]:
            mertens[i].set_color(RED)
        mertens[-1].set_color(YELLOW)
        mertens[6].set_color(BLUE)

        mertens.shift(UP*2)

        mertens_title = TextMobject("The ", r"""Mertens function """, "is given as,")
        mertens_title[1].set_color(GREEN)
        mertens_title.shift(UP * 3.5)
        
        primes = "distinct primes"
        loc = str(mobius[5]).find(primes)

        self.play(
            Write(mobius_title),
            FadeInFrom(mobius, UP))
        self.wait()
        self.play(Indicate(mobius[5][27 + loc:25 + len(primes)]))
        self.wait()

        mobius.generate_target()
        mobius.target.shift(DOWN * 2)

        self.play(
            MoveToTarget(mobius),
            LaggedStart(*map(FadeOutAndShiftDown, mobius_title)))
        self.wait()
        self.play(
            Write(mertens_title),
            Write(mertens[0:7]),
            ReplacementTransform(mobius[0:4].copy(), mertens[7]))
        self.wait(2)

        transform_mobius = mobius.copy()
        transform_mobius.to_corner(UR)
        transform_mobius.shift(RIGHT*2 + UP * .75)
        transform_mobius.scale(.6)

        transform_mertens = mertens.copy()
        transform_mertens.shift(LEFT*3)

        # mertens.generate_target()
        # mertens.target.shift(LEFT*3)

        self.play(
            Transform(mobius, transform_mobius),
            Transform(mertens[0:8], transform_mertens),
            LaggedStart(*map(FadeOutAndShiftDown, mertens_title))
            )

        cts_mert = TexMobject("M(", "x", ")", "=", "M", "(", r"\lfloor x \rfloor", "), \quad ", "x", "\in", "\R")
        cts_mert.match_x(mertens[0:3], direction=LEFT)
        for i in [0, 2, 4, 5, 7]:
            cts_mert[i].set_color(GREEN)
        cts_mert[7][1].set_color(WHITE)
        for i in [1, 6, 8]:
            cts_mert[i].set_color(RED)
        cts_mert[10].set_color(YELLOW)

        self.play(
            Transform(mertens[0:4].copy(), cts_mert[0:4]),
            Transform(mertens[0:4].copy(), cts_mert[4:8]),
            run_time=2)
        self.play(Transform(transform_mertens[8:].copy(), cts_mert[8:]))
        self.wait(2)
        self.play(Write(cts_mert))

        conjec = TexMobject("|", "M(", "x", ")", "|", "<", "\sqrt{x}")
        conjec.match_x(mertens[0:3], direction=LEFT)
        conjec.shift(DOWN)
        for i in [1, 3]:
            conjec[i].set_color(GREEN)
        for i in [2, -1]:
            conjec[i].set_color(RED)

        self.play(
            Transform(cts_mert[0:4].copy(), conjec[0:5]))
        self.play(Write(conjec))


class GraphMert(GraphScene):
    CONFIG={
        "x_min": 0,
        "x_max": 100,
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
    