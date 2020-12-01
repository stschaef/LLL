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
        
        self.play(Write(conjec))

        # TODO: Connection to RH and LLL


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
    