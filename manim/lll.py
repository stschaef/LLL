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

class GramSchmidtAnim(VectorScene):
    GS_COLOR = RED

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
        my_dict = {
            # "x_min": 0,
            # "x_max": 30,
            # "y_min": -5,
            # "y_max": 20
        }
        self.plane = self.add_plane(animate=True, **my_dict)
        self.axes = self.add_axes(animate=True, **my_dict)

        colors = [YELLOW, GREEN, RED, BLUE]

        self.basis = np.array([[1, 2],
                          [5, -3]])

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
        
        self.send_to_GS_basis(self.basis[0],
                              self.basis_objs[0],
                              label=self.basis_labels[0],
                              coords=self.basis_coords[0])
        
        for i in range(1, len(self.basis)):
            for j in range(i):
                self.mu[i][j] = np.dot(self.basis[i], self.orthog[j])/sq_norm(self.orthog[j])

            # Working vector
            v = self.basis[i]
            self.play(Indicate(self.basis_objs[i]))
            self.play(FadeOut(self.basis_labels[i]))

            # TODO: Show projections and subtract them off 
            for j in range(i):
                proj = self.mu[i][j] * self.orthog[j]
                self.add_vector(proj, color=TEAL, animate=True)
                v = v - proj

            # TODO: Solidify this as part of GS basis
            # orthog[i] = v
        





# print(gram_schmidt(np.array([[1, 2, 3, 0],
#                              [1, 2, 0, 0],
#                              [1, 0, 0, 1]])))

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