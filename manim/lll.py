from manimlib.imports import *
import numpy as np

# TODO: Opening Quote
# TODO: Outline of video
# TODO: LLL Definitions (Size and Lovasz)
# TODO: LLL psuedocode
# TODO: Approxs of alg numbers
# TODO: Crypto
# TODO: Rework Mertens stuff
# TODO: Number Field stuff 

BASIS_COLOR = BLUE
GS_COLOR = RED
MU_COLOR = GREEN
HIGHLIGHT_COLOR = YELLOW


class Psuedocode(Scene):
    def basis_str(self, n):
        return r"\textbf{b}_" + str(n)
    def gs_str(self, n):
        return r"\textbf{b}^*_" + str(n)
    def construct(self):
        def set_colors_and_scale(obj):
            obj.set_color_by_tex(r"\textbf{b}", BASIS_COLOR)
            obj.set_color_by_tex(r"\textbf{b}^*", GS_COLOR)
            obj.set_color_by_tex(r"\mu", MU_COLOR)
            obj.scale(.8)

        given = TexMobject(r"\text{Lattice }", "L", r"\text{ with basis }", self.basis_str(0) + r",\dots," + self.basis_str("n"))
        set_colors_and_scale(given)
        given[1].set_color(TEAL)
        given.to_corner(UL)
        self.play(Write(given))

        self.wait(2)

        gs = TexMobject(r"\text{Gram-Schmidt}" ,r"\text{ vectors }", self.gs_str(0) + r",\dots," + self.gs_str("n"))
        set_colors_and_scale(gs)
        gs.next_to(given, direction=DOWN)
        gs.to_edge(LEFT)
        self.play(Write(gs))

        self.wait(2)

        mu_intro = TexMobject(r"\text{Gram-Schmidt Coefficients, } ", r"\mu_{i, j}", "= {", self.basis_str("i"), r" \cdot", self.gs_str("j"), r"\over", self.gs_str("j"), r" \cdot", self.gs_str("j"), r"}")
        set_colors_and_scale(mu_intro)
        mu_intro.next_to(gs, direction=DOWN)
        mu_intro.to_edge(LEFT)
        
        self.play(Write(mu_intro))

        self.wait(2)

        k_list = [r"\text{Index }", r"\text{of working vector }", r"k"]
        k_tex = TexMobject(*k_list)
        k_tex[-1].set_color(HIGHLIGHT_COLOR)
        set_colors_and_scale(k_tex)
        k_tex.next_to(mu_intro, direction=DOWN)
        k_tex.to_edge(LEFT)

        self.play(Write(k_tex))

        self.wait(2)

        idx_list = [k_list[0], k_list[-1]]
        idx_k = TexMobject(*idx_list)
        idx_k[-1].set_color(HIGHLIGHT_COLOR)
        set_colors_and_scale(idx_k)
        idx_k.move_to(k_tex)


        self.play(ReplacementTransform(k_tex, idx_k))


        basis = copy.deepcopy(given[-1])
        gs_vecs = copy.deepcopy(gs[-1])
        mu_ij = copy.deepcopy(mu_intro[1:])

        self.add(basis)
        self.add(gs_vecs)
        self.add(mu_ij)

        self.play(FadeOut(gs), FadeOut(given), FadeOut(mu_intro))

        basis.generate_target()
        basis.target.to_corner(UL)
        
        gs_vecs.generate_target()
        gs_vecs.target.next_to(basis.target, direction=RIGHT, buff=2)

        mu_ij.generate_target()
        mu_ij.target.next_to(gs_vecs.target, direction=RIGHT, buff=2)

        idx_k.generate_target()
        idx_k.target.next_to(mu_ij.target, direction=RIGHT, buff=2)

        self.play(*[MoveToTarget(a) for a in [basis, gs_vecs, mu_ij, idx_k]])


        

        

# TODO:Work out colors
class LLLSymbolic2D(Scene):
    def make_vec(self, coords, color_in=WHITE):
        coord_str = ""
        for i in range(self.n_dim):
            if i != self.n_dim - 1:
                coord_str += str(coords[i]) + r" \\ "
            else:
                coord_str += str(coords[i])
        return TexMobject(r"\begin{bmatrix} " + coord_str + r" \end{bmatrix}", color=color_in)


    def init_vars(self, basis=None):
        self.delta = .75

        self.basis = np.array([[47, 215, 0],
                               [95, 460, 0]])
        if basis is not None:
            self.basis = basis
        else:
            self.n_dim = 2

        self.basis_text = TextMobject("Basis")
        self.basis_text.to_corner(UL, buff=1)

        self.basis_objs = [self.make_vec(b, color_in=BASIS_COLOR) for i, b in enumerate(self.basis)]
        self.basis_labels = [TexMobject(r"\textbf{b}_" + str(i) + " = ", color=BASIS_COLOR) for i, _ in enumerate(self.basis)]
        self.basis_displays = [None for _ in self.basis]
        for i, obj in enumerate(self.basis_objs):
            self.basis_labels[i].next_to(obj, direction=LEFT)
            self.basis_displays[i] = VGroup(self.basis_labels[i], obj)
            if i == 0:
                self.basis_displays[i].next_to(self.basis_text, RIGHT, buff=3/self.n_dim)
            else:
                self.basis_displays[i].next_to(self.basis_displays[i - 1], RIGHT, buff=3/self.n_dim)

        self.play(Write(self.basis_text))
        self.play(*[Write(o) for o in self.basis_displays])

        self.gs_text = TextMobject(r"Gram- \\ Schmidt")
        self.gs_text.next_to(self.basis_text, direction=DOWN, buff=self.n_dim/2)

        self.orthog = self.gram_schmidt(copy.deepcopy(self.basis))
        self.update_mus()
        
        self.gs_objs = [self.make_vec(np.around(b, decimals=2), color_in=GS_COLOR) for i, b in enumerate(self.orthog)]
        self.gs_labels = [TexMobject(r"\textbf{b}^*_" + str(i) + " = ", color=GS_COLOR) for i, _ in enumerate(self.orthog)]
        self.gs_displays = [None for _ in self.orthog]
        for i, obj in enumerate(self.gs_objs):
            self.gs_labels[i].next_to(obj, direction=LEFT)
            self.gs_displays[i] = VGroup(self.gs_labels[i], obj)
            self.gs_displays[i].move_to((self.basis_displays[i].get_x(), self.gs_text.get_y(), 0))
            # , self.basis  (, direction=DOWN, buff=1)

        self.play(Write(self.gs_text))
        self.play(*[Write(o) for o in self.gs_displays])

        self.play(*[FadeOut(a) for a in [self.basis_text, self.gs_text]])
        a = VGroup(*self.basis_displays, *self.gs_displays)
        a.generate_target()
        a.target.to_edge(LEFT)
        self.play(MoveToTarget(a))

        r = SurroundingRectangle(a, color=WHITE, buff=.19)
        self.play(ShowCreation(r))

        self.k = 1
        self.k_tex = TexMobject("k", "=", str(self.k))
        self.k_tex.set_color(HIGHLIGHT_COLOR)
        self.k_tex.set_y(a.get_y())
        self.k_tex.scale(2)
        self.k_tex.to_edge(RIGHT)
        self.play(Write(self.k_tex))

    def update_k_tex(self):
        self.dehehighlight_working_vector()
        a = TexMobject("k", "=", str(self.k))
        a.set_color(HIGHLIGHT_COLOR)
        a.move_to(self.k_tex.get_center())
        a.scale(2)
        self.play(Transform(self.k_tex, a))

    def highlight_working_vector(self):
        self.basis_displays[self.k].generate_target()
        self.basis_displays[self.k].target.set_color(HIGHLIGHT_COLOR)
        self.play(MoveToTarget(self.basis_displays[self.k]))

        self.basis_objs[self.k].set_color(HIGHLIGHT_COLOR)

    def dehehighlight_working_vector(self):
        # Cheat and dehilight all vectors
        for i, _ in enumerate(self.basis):
            self.basis_displays[i].generate_target()
            self.basis_displays[i].target.set_color(BASIS_COLOR)
            self.basis_objs[i].set_color(BASIS_COLOR)

        self.play(*[MoveToTarget(self.basis_displays[i]) for i, _ in enumerate(self.basis)])

    def sq_norm(self, v):
        return np.dot(v, v)

    def gram_schmidt(self, basis):
        """Returns the Gram-Schmidt orthogonalization of a basis.

        basis: list of linearly independent vectors
        """
        orthog = np.array([None for _ in basis])
        mu = np.array([[None for _ in basis[0]] for _ in basis])
        
        orthog[0] = basis[0]

        for i in range(1, len(basis)):
            for j in range(i):
                mu[i][j] = np.dot(basis[i], orthog[j])/self.sq_norm(orthog[j])
            orthog[i] = basis[i]
            for j in range(i):
                orthog[i] = orthog[i] - mu[i][j] * orthog[j]
        return np.array(orthog)

    def update_mus(self):
        assert(len(self.basis) == len(self.orthog))
        mu = np.array([[None for _ in self.basis] for _ in self.basis])

        for i, b_i in enumerate(self.basis):
            for j, b_j_star in enumerate(self.orthog):
                mu[i][j] = np.dot(b_i, b_j_star)/self.sq_norm(b_j_star)

        self.mu = mu

    def update_gs(self):
        a = self.gram_schmidt(copy.deepcopy(self.basis))
        if not np.array([np.isclose(a_elt, o_elt) for (a_elt, o_elt) in zip(a, self.orthog)]).all():
            self.orthog = self.gram_schmidt(copy.deepcopy(self.basis))
            self.play(*[Indicate(a, color=MAROON) for a in self.gs_displays])
            self.wait(1)
            for i, b_star in enumerate(self.orthog):
                self.update_vector(self.gs_objs[i], b_star)
        else:
            self.orthog = self.gram_schmidt(copy.deepcopy(self.basis))
        self.update_mus()

    def update_vector(self, old_vector, new_coords):
        new_coords = np.around(new_coords, decimals=2)
        x = self.make_vec(new_coords, color_in=old_vector.get_color())
        x.set_color(old_vector.get_color())
        x.move_to(old_vector.get_center())
        self.play(Transform(old_vector, x))

    def set_obj_colors_by_tex(self, obj):
        obj.set_color_by_tex(r"\textbf{b}", BASIS_COLOR)
        obj.set_color_by_tex(r"\textbf{b}^*", GS_COLOR)
        obj.set_color_by_tex(r"\textbf{b}_" + str(self.k), HIGHLIGHT_COLOR)
        obj.set_color_by_tex(r"\mu", MU_COLOR)

    def size_condition_anim(self, cur):
        size = TextMobject(r"Size \\ Condition")
    
        mu_tex = TexMobject(r"|\mu_{" + str(self.k) + r"," + str(cur) + r"}|", "=", r"\bigg|", r"{ \textbf{b}_" + str(self.k), r"\cdot", r"\textbf{b}^*_" + str(cur), r" \over ", r"\textbf{b}^*_" + str(cur), r"\cdot", r"\textbf{b}^*_" + str(cur), r"} \bigg| = ")
        self.set_obj_colors_by_tex(mu_tex)
        
        mu_val = np.abs(np.around(self.mu[self.k][cur], decimals=2))
        mu_val_tex = TexMobject(mu_val)

        mu_val_tex.next_to(mu_tex, direction=RIGHT)

        mu_display = VGroup(mu_tex, mu_val_tex)
        mu_display.set_x(0)
        mu_display.shift(DOWN*1.1)

        size.next_to(mu_display, direction=LEFT, buff=2)

        self.play(Write(mu_display), Write(size))
        self.wait(1)
        self.play(FadeOutAndShiftDown(size))

        less_than = TextMobject(" < .5")
        less_than.next_to(mu_display)
        self.play(Write(less_than))
        
        # Size condition
        if mu_val > .5:
            xmark = TexMobject(r"\text{\sffamily X}")
            xmark.scale(2)
            xmark.next_to(less_than, direction=RIGHT)
            xmark.set_color(RED)
            self.play(Write(xmark))

            basis_before_change = copy.deepcopy(self.basis[self.k])
            self.basis[self.k] = self.basis[self.k] - np.rint(self.mu[self.k][cur]) * self.basis[cur]

            tex_list = [r"\textbf{b}_" + str(self.k) + r"}", " = ", r"\textbf{b}_" + str(self.k) + r"}", "-", r"\lfloor \mu_{" + str(self.k) + "," + str(cur) + r"} \rceil", r"\cdot", r"\textbf{b}_" + str(cur)]
    
            update_basis = TexMobject(*tex_list)
            update_basis.next_to(mu_display, direction=DOWN, buff=1)
            self.set_obj_colors_by_tex(update_basis)
            if cur == self.k:
                update_basis[-1].set_color(HIGHLIGHT_COLOR)

            color_dict = {}
            for i, mobj in enumerate(update_basis):
                color_dict[i] = mobj.get_color()

            update_basis.set_x(0)
            self.play(Write(update_basis))

            coords_k = self.make_vec(basis_before_change)
            a_list = copy.deepcopy(tex_list)
            a_list[2] = coords_k.get_tex_string()
            a = TexMobject(*a_list)
            for i, mobj in enumerate(a):
                mobj.set_color(color_dict[i])
            a.move_to(update_basis.get_center())
            self.play(ReplacementTransform(update_basis, a))
        
            self.wait(1)

            coords_cur = self.make_vec(self.basis[cur])
            b_list = copy.deepcopy(a_list)
            b_list[-1] = coords_cur.get_tex_string()
            b = TexMobject(*b_list)
            for i, mobj in enumerate(b):
                mobj.set_color(color_dict[i])
            a[-1].set_color(BASIS_COLOR)
            if cur == self.k:
                a[-1].set_color(HIGHLIGHT_COLOR)
            b.move_to(update_basis.get_center())
            self.play(ReplacementTransform(a, b))

            self.wait(1)

            unrounded = TexMobject(r"\lfloor" + str(np.around(self.mu[self.k][cur], decimals=2)) + r"\rceil")
            c_list = copy.deepcopy(b_list)
            c_list[-3] = unrounded.get_tex_string()
            c = TexMobject(*c_list)
            for i, mobj in enumerate(c):
                mobj.set_color(color_dict[i])
            c[-3].set_color(MU_COLOR)
            c.move_to(update_basis.get_center())
            self.play(ReplacementTransform(b, c))

            self.wait(1)

            rounded = TexMobject(str(int(np.rint(self.mu[self.k][cur]))))
            d_list = copy.deepcopy(c_list)
            d_list[-3] = rounded.get_tex_string()
            d = TexMobject(*d_list)
            for i, mobj in enumerate(d):
                mobj.set_color(color_dict[i])
            d[-3].set_color(MU_COLOR)
            d.move_to(update_basis.get_center())
            self.play(ReplacementTransform(c, d))
            
            equal = TexMobject("=")
            result = self.make_vec(self.basis[self.k], color_in=color_dict[0])
            result.next_to(equal, direction=RIGHT)
            result_group = VGroup(equal, result)
            result_group.next_to(d, direction=RIGHT)
            self.play(Write(result_group))

            self.update_vector(self.basis_objs[self.k], self.basis[self.k])
            self.play(FadeOut(d), FadeOut(result_group))
            self.play(FadeOut(mu_display), FadeOut(xmark), FadeOut(less_than))
            self.update_gs()
        else:
            check = TexMobject(r"\checkmark")
            check.scale(2)
            check.next_to(less_than, direction=RIGHT)
            check.set_color(GREEN)
            self.play(Write(check))
            self.wait(1)
            self.play(FadeOut(mu_display), FadeOut(check), FadeOut(less_than))


    def lovasz_anim(self):
        lov = TextMobject(r"Lov\'asv \\ Condition")
        lov.shift(DOWN)
        lov.set_x(0)

        lhs_tex = TexMobject(r"\left| \textbf{b}^*_" + str(self.k) + r" \right|^2")
        rhs_tex = TexMobject(r"\left( 3/4 -", r"\mu_{" + str(self.k) + "," + str(self.k - 1) + r"}", r"\right)^2 \cdot", r" \left| \textbf{b}^*_" + str(self.k - 1) + r" \right|^2")
        self.set_obj_colors_by_tex(lhs_tex)
        self.set_obj_colors_by_tex(rhs_tex)

        lhs = self.sq_norm(self.orthog[self.k])
        rhs = (self.delta - self.mu[self.k][self.k - 1]**2) * self.sq_norm(self.orthog[self.k - 1])

        geq = TexMobject(r"\geq")
        equal_1 = TexMobject(r"=")
        equal_2 = TexMobject(r"=")
        
        lhs_val_tex = TexMobject(str(np.around(lhs, decimals=2)))
        rhs_val_tex = TexMobject(str(np.around(rhs, decimals=2)))

        equal_1.next_to(lhs_tex)
        lhs_val_tex.next_to(equal_1)
        geq.next_to(lhs_val_tex)
        rhs_val_tex.next_to(geq)
        equal_2.next_to(rhs_val_tex)
        rhs_tex.next_to(equal_2)

        a = VGroup(lhs_tex, rhs_tex, lhs_val_tex, rhs_val_tex, equal_1, equal_2, geq)
        a.next_to(lov, direction=DOWN)

        self.play(Write(lov), ShowCreation(a))
        self.wait(1)
        self.play(FadeOutAndShift(lov))

        a.generate_target()
        a.target.shift(UP)
        self.play(MoveToTarget(a))

        if lhs >= rhs:
            check = TexMobject(r"\checkmark")
            check.scale(2)
            check.next_to(a, direction=RIGHT)
            check.set_color(GREEN)
            self.play(Write(check))
            self.wait(1)

            increment_k = TexMobject(r"\text{Increment }", r"k")
            increment_k[1].set_color(HIGHLIGHT_COLOR)
            increment_k.next_to(a, direction=DOWN)
            self.play(Write(increment_k))
            self.wait(2)
            self.k += 1
            self.update_k_tex()
            self.play(FadeOutAndShift(check), FadeOutAndShift(a), FadeOut(increment_k))
        else:
            xmark = TexMobject(r"\text{\sffamily X}")
            xmark.scale(2)
            xmark.next_to(a, direction=RIGHT)
            xmark.set_color(RED)
            self.play(Write(xmark))
            

            swap = TexMobject(r"\text{Swap}(", r"\textbf{b}_" + str(self.k), r",",  r"\textbf{b}_" + str(self.k - 1), r")")
            self.set_obj_colors_by_tex(swap)
            max_k_list = [r"k" , r"= \text{max}(", "k - 1", ", 1)"]
            max_k = TexMobject(*max_k_list)
            max_k[0].set_color(HIGHLIGHT_COLOR)
            swap.next_to(a, direction=DOWN)
            max_k.next_to(swap, direction=DOWN)
            self.play(Write(swap), Write(max_k))

            self.wait(2)


            k_minus_1 = str(self.k - 1)
            b_list = copy.deepcopy(max_k_list)
            b_list[2] = k_minus_1
            b = TexMobject(*b_list)
            b[0].set_color(HIGHLIGHT_COLOR)
            b.move_to(max_k.get_center())
            self.play(ReplacementTransform(max_k, b))

            self.wait(1)
            
            temp = copy.deepcopy(self.basis[self.k])
            self.basis[self.k] = copy.copy((self.basis[self.k - 1]))
            self.update_vector(self.basis_objs[self.k], self.basis[self.k])
            self.basis[self.k - 1] = temp
            self.update_vector(self.basis_objs[self.k - 1], temp)

            self.update_gs()
            self.k = max(self.k - 1, 1)
            self.update_k_tex()

            self.play(FadeOutAndShift(a), FadeOutAndShift(xmark), FadeOutAndShift(swap), FadeOutAndShift(b))


    def lll_anim(self):
        self.update_gs()
        while self.k < len(self.basis):
            self.highlight_working_vector()
            for j in range(self.k - 1, -1, -1):
                self.size_condition_anim(j)
            self.lovasz_anim()
                

    def lll(self, basis, delta=.75):
        """Returns an LLL-reduced basis.

        basis: list of linearly independent vectors
        delta: commonly delta = 3/4
        """
        orthog = self.gram_schmidt(basis)
        mu = get_mus(basis, orthog)
        k = 1
        while k < len(basis):
            for j in range(k - 1, -1, -1):
                # Size condition
                if np.abs(mu[k][j]) > .5:
                    basis[k] = basis[k] - np.rint(mu[k, j]) * basis[j]
                    orthog = self.gram_schmidt(basis)
                    mu = self.get_mus(basis, orthog)
            # Lovasz condition
            if self.sq_norm(orthog[k]) >= (delta - mu[k][k - 1]**2) * self.sq_norm(orthog[k - 1]):
                k += 1
            else:
                temp = copy.deepcopy(basis[k])
                basis[k] = copy.deepcopy(basis[k - 1])
                basis[k - 1] = temp
                orthog = self.gram_schmidt(basis)
                mu = self.get_mus(basis, orthog)
                k = max(k - 1, 1)
        return basis


    def construct(self):
        self.init_vars()
        self.lll_anim()
        
class LLLSymbolic3D(LLLSymbolic2D):
    def construct(self):
        self.n_dim = 3
        self.init_vars(basis=np.array([[1, 1, 1],
                                       [-1, 0, 2],
                                       [3, 5, 6]]))
        self.lll_anim()
    
        
        

        


        

        
# def polynomial_string(poly_list):
#     out = ""
#     for i, coeff in enumerate(poly_list):
#         coeff = int(coeff)
#         if coeff != 0:
#             sign = "+"
#             if coeff < 0:
#                 sign = "-"
#             if i == 0:
#                 out += str(coeff)
#                 continue
#             coeff_str = abs(coeff)
#             if abs(coeff) == 1:
#                 coeff_str = ""
#             out += " {} {}x^{}".format(sign, coeff_str, i)
#     return out

def minpoly(alpha, deg, prec=10**6):
    A = list(np.identity(deg + 1))
    A[0] = np.concatenate((A[0], [prec]), axis=None)
    for i in range(1, deg + 1):
        A[i] = np.concatenate((A[i], [np.floor(prec * alpha**i)]), axis=None)

    # A.append(np.zeros(len(A[0])))
    A = np.array(A)
    B = lll(A)
    b_1 = A[0][:-1]
    
    print(b_1)
    print(polynomial_string(b_1))

    return b_1

def check(poly_list, alpha):
    return sum([coeff * alpha**i for i, coeff in enumerate(poly_list)])
        

# a = 1.348006154
# b = minpoly(a, 6, prec=10**(len(str(a)) - 2)) 
# print(check(b, a))      
        