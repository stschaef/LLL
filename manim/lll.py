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


class LatticeAnim(VectorScene, ThreeDScene):
    def create_lattice(self):
        max_elt = max(max(self.u_orig.flatten()), max(self.v_orig.flatten()))
        bound = int(self.factor / max_elt) * 8
        # print(bound)
        # exit(1)

        self.lattice = []

        for i in range(-bound, bound, 5):
            for j in range(-bound, bound, 5):
                self.lattice.append(Dot(i*self.u + j*self.v, radius=0.02, fill_opactiy=0.3))

        self.play(*[ShowCreation(p) for p in self.lattice])

    def real_coords(self, vect, vect_obj, color_in, int_lbls=False):
        real_vect = self.get_vector(vect * self.factor)
        return vector_coordinate_label(real_vect, color=color_in ,integer_labels=int_lbls)


    def construct(self):
        colors = [YELLOW, GREEN, BLUE]
        self.basis_orig = np.array([[201,37,0],
                              [1648,297,0]])

        self.plane = self.add_plane(animate=True)
        self.u_orig, self.v_orig = np.array([40,1,0]), np.array([1,32,0])
        self.factor = max(self.basis_orig.flatten()) / 5
        self.u, self.v = self.u_orig / self.factor, self.v_orig / self.factor
        self.create_lattice()

        self.basis = self.basis_orig / self.factor
        self.basis_objs = [None for _ in self.basis]

        for i, b in enumerate(self.basis):
            self.basis_objs[i] = self.add_vector(b, color=colors[i], animate=True)
        
        self.basis_labels = [TexMobject(r"\textbf{b}_" + str(i)) for i, _ in enumerate(self.basis)]
        for i, label in enumerate(self.basis_labels):
            label.set_color(colors[i])
            self.label_vector(self.basis_objs[i], label)
        
        self.basis_coords = [self.real_coords(b, self.basis_objs[i], colors[i], int_lbls=True) for i, b in enumerate(self.basis)]

        for i, coord in enumerate(self.basis_coords):
            label = copy.deepcopy(self.basis_labels[i])

            if i == 0:
                coord.to_corner(UL)
            else:
                coord.next_to(self.basis_coords[i - 1])
            label.next_to(coord, direction=DOWN)
            self.add_fixed_in_frame_mobjects(coord)
            self.play(ShowCreation(coord), ShowCreation(label))

        self.lll_anim(self.basis)

    def gram_schmidt_vectors(self, basis):
        orthog = gram_schmidt(basis)
        orthog_objs = [self.add_vector(b, color=RED, animate=True) for b in orthog]
        return orthog, orthog_objs

    def lll_anim(self, basis, delta=.75):
        """Returns an LLL-reduced basis.

        basis: list of linearly independent vectors
        delta: commonly delta = 3/4
        """
        colors = [YELLOW, GREEN, BLUE]
        self.orthog, self.orthog_objs = self.gram_schmidt_vectors(self.basis)
        mu = get_mus(self.basis, self.orthog)
        k = 1
        while k < len(basis):
            for j in range(k - 1, -1, -1):
                # Size condition
                if np.abs(mu[k][j]) > .5:
                    self.basis[k] = self.basis[k] - np.rint(mu[k, j]) * self.basis[j]
                    self.play(ReplacementTransform(self.basis_objs[k],
                                                   self.get_vector(self.basis[k], color=colors[k])))
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

        

        # def scaleDot(dot, factor):
        #     return Dot(dot.arc_center * factor, radius=0.02, fill_opactiy=0.5)
        # self.play(*[ReplacementTransform(p, scaleDot(p, 2)) for p in self.lattice])
        


            
        