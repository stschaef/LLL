from manimlib.imports import *
import numpy as np

def sq_norm(v):
    return np.dot(v, v)

def gram_schmidt(basis):
    """Returns the Gram-Schmidt orthogonalization of a basis.

    basis: list of linearly independent vectors
    """
    orthog = np.array([None for _ in basis])
    mu = np.array([[None for _ in basis] for _ in basis])
    
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
            print(sq_norm(b_j_star))
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

class Plane3D(ParametricSurface):
    CONFIG = {
        "resolution": (8, 8),
        "u_min": -2,
        "u_max": 2,
        "v_min": -2,
        "v_max": 2,
        "vec_1": None,
        "vec_2": None,
        "fill_opacity": 0.2,
    }

    def __init__(self, **kwargs):
        ParametricSurface.__init__(
            self, self.func, **kwargs
        )

    def func(self, u, v):
        return self.vec_1 * u + self.vec_2 * v

class test(ThreeDScene):

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

    # np.array([[1, 1, 1],
    #           [-1, 0, 2],
    #           [3, 5, 6]])

    def construct(self):
        axes = ThreeDAxes()

        p = Plane3D(vec_1=np.array([0,0,2]), vec_2=np.array([1,2,3]))
        self.set_camera_orientation(phi=80*DEGREES, theta=45*DEGREES)
        self.play(ShowCreation(p), ShowCreation(axes))
        self.begin_ambient_camera_rotation(rate=2)
        self.wait(9)

class GramSchmidt2D(VectorScene):
    GS_COLOR = RED
    colors = [YELLOW, GREEN, RED, BLUE, PURPLE]


    def initialize_stuff(self):
        self.basis = np.array([[1, 2],
                               [4, -1]])

        my_dict = {
            "x_min": min(self.basis.T[0]) + 1,
            "x_max": max(self.basis.T[0]) + 1,
            "y_min": min(self.basis.T[1]) + 1,
            "y_max": max(self.basis.T[1]) + 1,
        }

        self.plane = self.add_plane(animate=True, **my_dict)
        self.axes = self.add_axes(animate=True, **my_dict)

        self.basis_objs = [self.add_vector(b, color=colors[i], animate=True) for i, b in enumerate(self.basis)]
        self.basis_labels = [TexMobject(r"\textbf{b}_" + str(i)) for i, _ in enumerate(self.basis)]
        
        for i, label in enumerate(self.basis_labels):
            label.set_color(colors[i])
            self.label_vector(self.basis_objs[i], label)

        self.wait(2)

        self.basis_coords = [self.write_vector_coordinates(b,
                                                           color=colors[i],
                                                           integer_labels=False) 
                                                           for i, b in enumerate(self.basis_objs)]

        self.orthog = np.array([None for _ in self.basis])
        self.orthog_objs = [None for _ in self.basis]
        self.orthog_coords = [None for _ in self.basis]
        self.orthog_labels = [TexMobject(r"\textbf{b}^*_" + str(i)) for i, _ in enumerate(self.orthog)]
        
        for label in self.orthog_labels:
            label.set_color(self.GS_COLOR)
        
        self.mu = np.array([[None for _ in self.basis] for _ in self.basis])

     
    def send_to_GS_basis(self, vect, vect_obj, label=None, coords=None):
        k = 0
        while self.orthog[k] is not None:
            k += 1
        self.orthog[k] = vect
        self.orthog_objs[k] = self.get_vector(vect, color=self.GS_COLOR)
        self.orthog_coords[k] = vector_coordinate_label(self.orthog_objs[k],
                                                        color=self.GS_COLOR,
                                                        integer_labels=False,
                                                        num_places=2)
        self.get_vector_label(self.orthog_objs[k], self.orthog_labels[k])

        tasks = []

        tasks.append(ReplacementTransform(vect_obj, self.orthog_objs[k]))
        if label is not None:
            tasks.append(ReplacementTransform(label, self.orthog_labels[k]))
        else:
            tasks.append(Write(self.orthog_labels[k]))
        if coords is not None:
            tasks.append(ReplacementTransform(coords, self.orthog_coords[k]))
        else:
            tasks.append(Write(self.orthog_coords[k]))
        
        self.play(*tasks, run_time=2)

    def construct(self):
        self.initialize_stuff()
        # Take first vector as part of orthogonal basis
        self.send_to_GS_basis(self.basis[0],
                              self.basis_objs[0],
                              label=self.basis_labels[0],
                              coords=self.basis_coords[0])
        
        for i in range(1, len(self.basis)):
            for j in range(i):
                self.mu[i][j] = np.dot(self.basis[i], self.orthog[j])/sq_norm(self.orthog[j])

            # Working vector
            v = self.basis[i]
            v_obj = self.basis_objs[i]
            v_coords = self.basis_coords[i]

            self.play(Indicate(self.basis_objs[i]))
            self.play(FadeOut(self.basis_labels[i]))

            for j in range(i):
                proj = self.mu[i][j] * self.orthog[j]
                a = v - proj
                proj_obj_origin = self.add_vector(proj, color=TEAL, animate=True)
                proj_obj = Arrow(np.array([v[0], v[1], 0]), np.array([a[0], a[1], 0]), color=TEAL, buff=0)

                self.play(ReplacementTransform(proj_obj_origin, proj_obj))
               
                v = v - proj
                subtracted_obj = self.get_vector(v, color=self.basis_objs[i].get_fill_color())

                subtracted_coords = vector_coordinate_label(subtracted_obj,
                                                            color=subtracted_obj.get_fill_color(),
                                                            integer_labels=False,
                                                            num_places=2)
                
                self.play(ReplacementTransform(v_obj, subtracted_obj),
                          ReplacementTransform(v_coords, subtracted_coords))
                self.wait(1)
                self.play(FadeOut(proj_obj))


            self.send_to_GS_basis(v, v_obj, coords=v_coords)


class GramSchmidt3D(ThreeDScene):
    GS_COLOR = RED

    def add_vector(self, vector, color=YELLOW, animate=True, **kwargs):
        if not isinstance(vector, Arrow):
            vector = Vector(vector, color=color, **kwargs)
        if animate:
            self.play(GrowArrow(vector))
        self.add(vector)
        return vector
    
    def write_vector_coordinates(self, vector, **kwargs):
        coords = vector_coordinate_label(vector, n_dim=3, **kwargs)
        self.play(Write(coords))
        return coords

    def get_vector_label(self, vector, label,
                         at_tip=False,
                         direction="left",
                         rotate=False,
                         color=None,
                         label_scale_factor=VECTOR_LABEL_SCALE_FACTOR):
        if not isinstance(label, TexMobject):
            if len(label) == 1:
                label = "\\vec{\\textbf{%s}}" % label
            label = TexMobject(label)
            if color is None:
                color = vector.get_color()
            label.set_color(color)
        label.scale(label_scale_factor)
        label.add_background_rectangle()

        if at_tip:
            vect = vector.get_vector()
            vect /= get_norm(vect)
            label.next_to(vector.get_end(), vect, buff=SMALL_BUFF)
        else:
            angle = vector.get_angle()
            if not rotate:
                label.rotate(-angle, about_point=ORIGIN)
            if direction == "left":
                label.shift(-label.get_bottom() + 0.1 * UP)
            else:
                label.shift(-label.get_top() + 0.1 * DOWN)
            label.rotate(angle, about_point=ORIGIN)
            label.shift((vector.get_end() - vector.get_start()) / 2)
        return label

    def label_vector(self, vector, label, animate=True, **kwargs):
        label = self.get_vector_label(vector, label, **kwargs)
        if animate:
            self.play(Write(label, run_time=1))
        self.add(label)
        return label

    def get_vector(self, numerical_vector, **kwargs):
        return Vector(numerical_vector, **kwargs)

    def intitialize_stuff(self):
        colors = [YELLOW, GREEN, BLUE, PURPLE]

        self.basis = np.array([[2, 2, 3],
                               [1, 2, -1],
                               [1, 0, 0]])

        my_dict = {
            "x_min": min(self.basis.T[0]) - 5,
            "x_max": max(self.basis.T[0]) + 5,
            "y_min": min(self.basis.T[1]) - 5,
            "y_max": max(self.basis.T[1]) + 5,
            "z_min": min(self.basis.T[2]) - 5,
            "z_max": max(self.basis.T[2]) + 5,
        }

        # self.plane = NumberPlane()
        self.axes = ThreeDAxes(**my_dict)
        self.set_camera_orientation(phi=80*DEGREES, theta=45*DEGREES)
        self.play(ShowCreation(self.axes)) 
        self.begin_ambient_camera_rotation(rate=.3)
        self.wait(3)
        
        self.basis_objs = [self.add_vector(b, color=colors[i], animate=True) for i, b in enumerate(self.basis)]
        self.basis_labels = [TexMobject(r"\textbf{b}_" + str(i)) for i, _ in enumerate(self.basis)]
        for i, label in enumerate(self.basis_labels):
            label.set_color(colors[i])
            self.label_vector(self.basis_objs[i], label)

        self.wait(2)

        self.basis_coords = [self.write_vector_coordinates(b,
                                                           color=colors[i],
                                                           integer_labels=False) 
                                                           for i, b in enumerate(self.basis_objs)]
        for coord in self.basis_coords:
            self.add_fixed_orientation_mobjects(coord)
        self.orthog = np.array([None for _ in self.basis])
        self.orthog_objs = [None for _ in self.basis]
        self.orthog_coords = [None for _ in self.basis]
        self.orthog_labels = [TexMobject(r"\textbf{b}^*_" + str(i)) for i, _ in enumerate(self.orthog)]
        
        for label in self.orthog_labels:
            label.set_color(self.GS_COLOR)
        
        self.mu = np.array([[None for _ in self.basis] for _ in self.basis])

     
    def send_to_GS_basis(self, vect, vect_obj, label=None, coords=None):
        k = 0
        while self.orthog[k] is not None:
            k += 1
        self.orthog[k] = vect
        self.orthog_objs[k] = self.get_vector(vect, color=self.GS_COLOR)
        self.orthog_coords[k] = vector_coordinate_label(self.orthog_objs[k],
                                                        color=self.GS_COLOR,
                                                        integer_labels=False,
                                                        num_places=2)
        self.get_vector_label(self.orthog_objs[k], self.orthog_labels[k])

        tasks = []

        tasks.append(ReplacementTransform(vect_obj, self.orthog_objs[k]))
        if label is not None:
            tasks.append(ReplacementTransform(label, self.orthog_labels[k]))
        else:
            tasks.append(Write(self.orthog_labels[k]))
        if coords is not None:
            tasks.append(ReplacementTransform(coords, self.orthog_coords[k]))
        else:
            tasks.append(Write(self.orthog_coords[k]))
            self.add_fixed_orientation_mobjects(self.orthog_coords[k])
        
        self.play(*tasks, run_time=2)

    def construct(self):
        self.intitialize_stuff()

        

        # Take first vector as part of orthogonal basis
        self.send_to_GS_basis(self.basis[0],
                              self.basis_objs[0],
                              label=self.basis_labels[0],
                              coords=self.basis_coords[0])
        
        for i in range(1, len(self.basis)):
            for j in range(i):
                self.mu[i][j] = np.dot(self.basis[i], self.orthog[j])/sq_norm(self.orthog[j])

            # Working vector
            v = self.basis[i]
            v_obj = self.basis_objs[i]
            v_coords = self.basis_coords[i]

            self.play(Indicate(self.basis_objs[i]))
            self.play(FadeOut(self.basis_labels[i]))

            for j in range(i):
                proj = self.mu[i][j] * self.orthog[j]
                a = v - proj
                proj_obj_origin = self.add_vector(proj, color=TEAL, animate=True)
                proj_obj = Arrow(np.array([v[0], v[1], v[2]]), np.array([a[0], a[1], v[2]]), color=TEAL, buff=0)

                self.play(ReplacementTransform(proj_obj_origin, proj_obj))
               
                v = v - proj
                subtracted_obj = self.get_vector(v, color=self.basis_objs[i].get_fill_color())

                subtracted_coords = vector_coordinate_label(subtracted_obj,
                                                            color=subtracted_obj.get_fill_color(),
                                                            integer_labels=False,
                                                            num_places=2)
                self.add_fixed_orientation_mobjects(subtracted_coords)
                
                self.play(ReplacementTransform(v_obj, subtracted_obj),
                          ReplacementTransform(v_coords, subtracted_coords))
                self.wait(1)
                self.play(FadeOut(proj_obj))


            self.send_to_GS_basis(v, v_obj, coords=v_coords)





# print(gram_schmidt(np.array([[2, 2, 3],
#                              [1, 2, -1],
#                              [1, 0, 0]])))

# Jeff Suzuki LLL Example 1
# https://www.youtube.com/watch?v=n5MfVR77BTw
# print(lll(np.array([[1, 1, 1],
#                     [-1, 0, 2],
#                     [3, 5, 6]])))

# Jeff Suzuku LLL Example 2
# https://www.youtube.com/watch?v=n5MfVR77BTw
# print(lll(np.array([[15, 23, 11],
#                     [46, 15, 3],
#                     [32, 1, 1]])))

# print(lll(np.array([[201, 37],
#                     [1648, 297]])))