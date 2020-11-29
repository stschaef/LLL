from manimlib.imports import *
import numpy as np

def sq_norm(v):
    return np.dot(v, v)

def gram_schmidt(basis):
    """Returns the Gram-Schmidt orthogonalization of a basis.

    basis: list of linearly independent vectors
    """
    orthog = np.array([None for _ in basis])
    mu = np.array([[None for _ in basis[0]] for _ in basis])
    
    orthog[0] = basis[0]

    for i in range(1, len(basis)):
        for j in range(i):
            mu[i][j] = np.dot(basis[i], orthog[j])/sq_norm(orthog[j])
        orthog[i] = basis[i]
        for j in range(i):
            orthog[i] = orthog[i] - mu[i][j] * orthog[j]
    return orthog

def get_mus(basis, orthog):
    assert(len(basis) == len(orthog))
    mu = np.array([[None for _ in basis] for _ in basis])

    for i, b_i in enumerate(basis):
        for j, b_j_star in enumerate(orthog):
            # print(sq_norm(b_j_star))
            mu[i][j] = np.dot(b_i, b_j_star)/sq_norm(b_j_star)

    return mu

def lll(basis, delta=.75):
    """Returns an LLL-reduced basis.

    basis: list of linearly independent vectors
    delta: commonly delta = 3/4
    """
    orthog = gram_schmidt(basis)
    mu = get_mus(basis, orthog)
    k = 1
    while k < len(basis):
        for j in range(k - 1, -1, -1):
            # Size condition
            if np.abs(mu[k][j]) > .5:
                basis[k] = basis[k] - np.rint(mu[k, j]) * basis[j]
                orthog = gram_schmidt(basis)
                mu = get_mus(basis, orthog)
        # Lovasz condition
        if sq_norm(orthog[k]) >= (delta - mu[k][k - 1]**2) * sq_norm(orthog[k - 1]):
            k += 1
        else:
            temp = copy.deepcopy(basis[k])
            basis[k] = copy.deepcopy(basis[k - 1])
            basis[k - 1] = temp
            orthog = gram_schmidt(basis)
            mu = get_mus(basis, orthog)
            k = max(k - 1, 1)
    return basis


# class LatticeAnim(VectorScene, ThreeDScene):
#     def create_lattice(self):
#         max_elt = max(max(self.u_orig.flatten()), max(self.v_orig.flatten()))
#         bound = int(self.factor / max_elt) * 8
#         # print(bound)
#         # exit(1)

#         self.lattice = []

#         for i in range(-bound, bound, 3):
#             for j in range(-bound, bound, 3):
#                 self.lattice.append(Dot(i*self.u + j*self.v, radius=0.02, fill_opactiy=0.3))

#         self.play(*[ShowCreation(p) for p in self.lattice])

#     def real_coords(self, vect, vect_obj, color_in, int_lbls=False):
#         real_vect = self.get_vector(vect * self.factor)
#         return vector_coordinate_label(real_vect, color=color_in ,integer_labels=int_lbls)


#     def construct(self):
#         self.colors = [YELLOW, GREEN, BLUE]
#         self.basis_orig = np.array([[201,37,0],
#                               [1648,297,0]])

#         self.plane = self.add_plane(animate=True)
#         self.u_orig, self.v_orig = np.array([40,1,0]), np.array([1,32,0])
#         self.factor = max(self.basis_orig.flatten()) / 5
#         self.u, self.v = self.u_orig / self.factor, self.v_orig / self.factor
#         self.create_lattice()

#         # self.wait(1)
#         # def scaleDot(dot, factor, color_in):
#         #     return Dot(dot.arc_center * factor, radius=0.02, fill_opactiy=0.5, color=color_in)
#         # self.play(*[Transform(p, scaleDot(p, 2, YELLOW)) for p in self.lattice])
#         # self.play(*[Transform(p, scaleDot(p, .5, WHITE)) for p in self.lattice])
#         # # self.play(*[Indicate(p) for p in self.lattice])
#         # self.wait(1)

#         self.basis = self.basis_orig / self.factor
#         self.basis_objs = [None for _ in self.basis]

#         for i, b in enumerate(self.basis):
#             self.basis_objs[i] = self.add_vector(b, color=self.colors[i], animate=True)
        
#         self.basis_labels = [TexMobject(r"\textbf{b}_" + str(i)) for i, _ in enumerate(self.basis)]
#         for i, label in enumerate(self.basis_labels):
#             label.set_color(self.colors[i])
#             self.label_vector(self.basis_objs[i], label)
        
#         self.basis_coords = [self.real_coords(b, self.basis_objs[i], self.colors[i], int_lbls=True) for i, b in enumerate(self.basis)]

#         for i, coord in enumerate(self.basis_coords):
#             label = copy.deepcopy(self.basis_labels[i])

#             if i == 0:
#                 coord.to_corner(UL)
#             else:
#                 coord.next_to(self.basis_coords[i - 1])
#             label.next_to(coord, direction=DOWN)
#             self.add_fixed_in_frame_mobjects(coord)
#             self.play(ShowCreation(coord), ShowCreation(label))

#         self.play(*[FadeOut(p) for p in self.lattice])

#         self.lll_anim(self.basis)

#     def gram_schmidt_vectors(self, basis):
#         orthog = gram_schmidt(basis)
#         orthog_objs = [self.add_vector(b, color=RED, animate=True) for b in orthog]
#         return orthog, orthog_objs

#     def lovasz_anim(self, delta=.75):
#         # sq_norm(self.orthog[self.k]) >= (delta - self.mu[self.k][self.k - 1]**2) * sq_norm(self.orthog[self.k - 1])
#         norm_sqrd = sq_norm(self.orthog[self.k])
#         lov_val = (delta - self.mu[self.k][self.k - 1]**2) * sq_norm(self.orthog[self.k - 1])

#         norm_tex_str = r"|| \textbf{b}^*_" + str(self.k) + r"||^2"
#         lov_tex_str = r"\left( 3/4 - \mu_{" + str(self.k) + "," + str(self.k - 1) + r"}^2 \right) || \textbf{b}^*_" + str(self.k - 1) + r"||^2"

#         norm_tex = TexMobject(norm_tex_str)
#         lov_tex = TexMobject(lov_tex_str)

#         norm_tex.to_corner(DL)
#         lov_tex.next_to(norm_tex)
#         lov_tex.shift(RIGHT)

#         self.play(ShowCreation(norm_tex), ShowCreation(lov_tex))

#     def update_lll_vars(self):
#         self.orthog = gram_schmidt(self.basis)
#         self.mu = get_mus(self.basis, self.orthog)

#     def swap_vects(self):
#         temp = copy.deepcopy(self.basis[self.k])
#         temp_obj = copy.deepcopy(self.basis_objs[self.k])
#         # temp_label = 
#         self.basis[self.k] = copy.deepcopy(self.basis[self.k - 1])
#         self.basis[self.k - 1] = temp

#         self.play(
#             ReplacementTransform(self.basis_objs[self.k], self.basis_objs[self.k - 1]),
#             ReplacementTransform(self.basis_objs[self.k - 1], temp_obj))
#         # self.basis_objs[self.k] = copy.deepcopy(self.basis_objs[self.k - 1])
#         # self.basis_objs[self.k - 1] = temp

#     def lll_anim(self, basis, delta=.75):
#         """Returns an LLL-reduced basis.

#         basis: list of linearly independent vectors
#         delta: commonly delta = 3/4
#         """
#         self.orthog, self.orthog_objs = self.gram_schmidt_vectors(self.basis)
#         self.mu = get_mus(self.basis, self.orthog)
#         self.k = 1
#         while self.k < len(self.basis):
#             for j in range(self.k - 1, -1, -1):
#                 # Size condition
#                 if np.abs(self.mu[self.k][j]) > .5:
#                     self.basis[self.k] = self.basis[self.k] - np.rint(self.mu[self.k, j]) * self.basis[j]
#                     self.play(ReplacementTransform(self.basis_objs[self.k],
#                                                    self.get_vector(self.basis[self.k], color=self.colors[self.k])))
#                     self.wait(2)
#                     self.update_lll_vars()
#             # Lovasz condition
#             self.lovasz_anim()
#             if sq_norm(self.orthog[self.k]) >= (delta - self.mu[self.k][self.k - 1]**2) * sq_norm(self.orthog[self.k - 1]):
#                 self.k += 1
#             else:
#                 self.swap_vects()
#                 self.update_lll_vars()
#                 self.k = max(self.k - 1, 1)
#         return basis

        

        
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

# TODO: Opening Quote
# TODO: Outline of video
# TODO: LLL Definitions (Size and Lovasz)
# TODO: LLL psuedocode
# TODO: Approxs of alg numbers
# TODO: Crypto
# TODO: Rework Mertens stuff
# TODO: Number Field stuff 

class LLLSymbolic2D(Scene):
    def make_vec(self, coords, color_in=WHITE):
        # TODO: Weird .0 at end of orthog vecs in the 2D case
        coord_str = ""
        for i, coord in enumerate(coords):
            if i != self.n_dim - 1:
                coord_str += str(coord) + r" \\ "
            else:
                coord_str += str(coord)
        return TexMobject(r"\begin{bmatrix} " + coord_str + r"\end{bmatrix}", color=color_in)


    def init_vars(self, basis=None):
        # self.colors = [YELLOW, GREEN, BLUE, PURPLE]
        self.n_dim = 2
        self.colors = [WHITE]*10
        self.delta = .75

        self.basis = np.array([[201, 37, 0],
                               [1648, 297, 0]])
        if basis is not None:
            self.basis = basis

        self.basis_text = TextMobject("Basis")
        self.basis_text.to_corner(UL, buff=1)

        self.basis_objs = [self.make_vec(b, color_in=self.colors[i]) for i, b in enumerate(self.basis)]
        self.basis_labels = [TexMobject(r"\textbf{b}_" + str(i) + " = ", color=self.colors[i]) for i, _ in enumerate(self.basis)]
        self.basis_displays = [None for _ in self.basis]
        for i, obj in enumerate(self.basis_objs):
            self.basis_labels[i].next_to(obj, direction=LEFT)
            self.basis_displays[i] = VGroup(self.basis_labels[i], obj)
            if i == 0:
                self.basis_displays[i].next_to(self.basis_text, RIGHT, buff=2/self.n_dim)
            else:
                self.basis_displays[i].next_to(self.basis_displays[i - 1], RIGHT, buff=2/self.n_dim)

        self.play(Write(self.basis_text))
        self.play(*[Write(o) for o in self.basis_displays])

        self.gs_text = TextMobject(r"Gram- \\ Schmidt")
        self.gs_text.next_to(self.basis_text, direction=DOWN, buff=self.n_dim/1.5)

        self.orthog = self.gram_schmidt(copy.deepcopy(self.basis))
        
        self.gs_objs = [self.make_vec(np.around(b, decimals=2), color_in=WHITE) for i, b in enumerate(self.orthog)]
        self.gs_labels = [TexMobject(r"\textbf{b}^*_" + str(i) + " = ", color=WHITE) for i, _ in enumerate(self.orthog)]
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

        r = SurroundingRectangle(a, color=RED)
        self.play(ShowCreation(r))

        self.k = 1
        self.k_tex = TexMobject("k", "=", str(self.k))
        self.k_tex.set_color(YELLOW)
        self.k_tex.set_y(a.get_y())
        self.k_tex.scale(2)
        self.k_tex.to_edge(RIGHT)
        self.play(Write(self.k_tex))

    def update_k_tex(self):
        self.dehehighlight_working_vector()
        a = TexMobject("k", "=", str(self.k))
        a.set_color(YELLOW)
        a.move_to(self.k_tex.get_center())
        a.scale(2)
        self.play(Transform(self.k_tex, a))

    def highlight_working_vector(self):
        self.basis_displays[self.k].generate_target()
        self.basis_displays[self.k].target.set_color(YELLOW)
        self.play(MoveToTarget(self.basis_displays[self.k]))

        self.basis_objs[self.k].set_color(YELLOW)

    def dehehighlight_working_vector(self):
        # Cheat and dehilight all vectors
        for i, _ in enumerate(self.basis):
            self.basis_displays[i].generate_target()
            self.basis_displays[i].target.set_color(WHITE)
            self.basis_objs[i].set_color(WHITE)

        self.play(*[MoveToTarget(self.basis_displays[i]) for i, _ in enumerate(self.basis)])

        # if self.k < len(self.basis):
        #     self.basis_displays[self.k].generate_target()
        #     self.basis_displays[self.k].target.set_color(WHITE)
        #     self.play(MoveToTarget(self.basis_displays[self.k]))

        #     self.basis_objs[self.k].set_color(WHITE)
        # else:
        #     self.basis_displays[self.k - 1].generate_target()
        #     self.basis_displays[self.k - 1].target.set_color(WHITE)
        #     self.play(MoveToTarget(self.basis_displays[self.k - 1]))

        #     self.basis_objs[self.k - 1].set_color(WHITE)

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
                mu[i][j] = np.dot(basis[i], orthog[j])/sq_norm(orthog[j])
            orthog[i] = basis[i]
            for j in range(i):
                orthog[i] = orthog[i] - mu[i][j] * orthog[j]
        return orthog

    def update_mus(self):
        assert(len(self.basis) == len(self.orthog))
        mu = np.array([[None for _ in self.basis] for _ in self.basis])

        for i, b_i in enumerate(self.basis):
            for j, b_j_star in enumerate(self.orthog):
                mu[i][j] = np.dot(b_i, b_j_star)/sq_norm(b_j_star)

        self.mu = mu

    def update_gs(self):
        self.orthog = self.gram_schmidt(self.basis)
        for i, b_star in enumerate(self.orthog):
            self.update_vector(self.gs_objs[i], b_star)
        self.update_mus()

    def update_vector(self, old_vector, new_coords):
        new_coords = np.around(new_coords, decimals=2)
        x = self.make_vec(new_coords)
        x.set_color(old_vector.get_color())
        x.move_to(old_vector.get_center())
        self.play(Transform(old_vector, x))

    def size_condition_anim(self, cur):
        size = TextMobject(r"Size \\ Condition")
        mu_tex = TexMobject(r"| \mu_{" + str(self.k) + "," + str(cur) + r"}| = \left|\frac{\textbf{b}_" + str(self.k) + r" \cdot \textbf{b}^*_" + str(cur) + r"}{\textbf{b}^*_" + str(cur) + r"\cdot \textbf{b}^*_" + str(cur) + r"}\right| = ")
        mu_val = np.abs(np.around(self.mu[self.k][cur], decimals=2))
        mu_val_tex = TexMobject(mu_val)

        mu_val_tex.next_to(mu_tex, direction=RIGHT)

        mu_display = VGroup(mu_tex, mu_val_tex)
        mu_display.set_x(0)
        mu_display.shift(DOWN)

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

            tex_list = [r"\textbf{b}_" + str(self.k) + r"} = ", r"\textbf{b}_" + str(self.k) + r"}", "-", r"\lfloor \mu_{" + str(self.k) + "," + str(cur) + r"} \rceil", r"\cdot", r"\textbf{b}_" + str(cur)]

            update_basis = TexMobject(*tex_list)
            update_basis.next_to(mu_display, direction=DOWN, buff=1)
            update_basis.set_x(0)
            self.play(Write(update_basis))

            coords_k = self.make_vec(basis_before_change)
            a_list = copy.deepcopy(tex_list)
            a_list[1] = coords_k.get_tex_string()
            a = TexMobject(*a_list)
            a.move_to(update_basis.get_center())
            self.play(ReplacementTransform(update_basis, a))
        
            self.wait(1)

            coords_cur = self.make_vec(self.basis[cur])
            b_list = copy.deepcopy(a_list)
            b_list[-1] = coords_cur.get_tex_string()
            b = TexMobject(*b_list)
            b.move_to(update_basis.get_center())
            self.play(ReplacementTransform(a, b))

            self.wait(1)

            unrounded = TexMobject(r"\lfloor" + str(np.around(self.mu[self.k][cur], decimals=2)) + r"\rceil")
            c_list = copy.deepcopy(b_list)
            c_list[-3] = unrounded.get_tex_string()
            c = TexMobject(*c_list)
            c.move_to(update_basis.get_center())
            self.play(ReplacementTransform(b, c))

            self.wait(1)

            rounded = TexMobject(str(int(np.rint(self.mu[self.k][cur]))))
            d_list = copy.deepcopy(c_list)
            d_list[-3] = rounded.get_tex_string()
            d = TexMobject(*d_list)
            d.move_to(update_basis.get_center())
            self.play(ReplacementTransform(c, d))
            
            equal = TexMobject("=")
            result = self.make_vec(self.basis[self.k])
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

        lhs_tex = TexMobject(r"\left| b^*_" + str(self.k) + r" \right|^2")
        rhs_tex = TexMobject(r"\left( 3/4 - \mu_{" + str(self.k) + "," + str(self.k - 1) + r"} \right)^2 \cdot \left| b^*_" + str(self.k - 1) + r" \right|^2")
        
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

            increment_k = TexMobject(r"\text{Increment } k")
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
            

            swap = TexMobject(r"\text{Swap}( \textbf{b}_" + str(self.k) + r", \textbf{b}_" + str(self.k - 1) + r")")
            max_k_list = [r"k = \text{max}(", "k - 1", ", 1)"]
            max_k = TexMobject(*max_k_list)
            swap.next_to(a, direction=DOWN)
            max_k.next_to(swap, direction=DOWN)
            self.play(Write(swap), Write(max_k))

            self.wait(2)


            k_minus_1 = str(self.k - 1)
            b_list = copy.deepcopy(max_k_list)
            b_list[1] = k_minus_1
            b = TexMobject(*b_list)
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
    def make_vec(self, coords, color_in=WHITE):
        return TexMobject(r"\begin{bmatrix} " + str(coords[0]) + r" \\ " + str(coords[1]) + r" \\ " + str(coords[2]) + r"\end{bmatrix}", color=color_in)
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
        