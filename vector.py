from math import sqrt


class InputError(Exception):
    pass


class WrongDimensionError(Exception):
    pass


class V(object):
    """Vector

    Raises
    ------
    InputError
        Wrong input type (cannot convert to Vector)
    TypeError
        Wrong parameter type

    Returns
    -------
    V
        Vector
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and type(args[0]) is tuple:
            self.coords = args[0]
        elif type(args[0]) is list:
            self.coords = tuple(args[0])
        elif type(args) is tuple:
            self.coords = args
        else:
            print('Could not convert {} to Vector!'.format(args))
            raise InputError()
        self.x = self.coords[0]
        if len(self.coords) > 1:
            self.y = self.coords[1]
            if len(self.coords) > 2:
                self.z = self.coords[2]
                if len(self.coords) > 3:
                    self.w = self.coords[3]

    def cast(self, cast_function):
        """Cast points coordinates to new type.

        Parameters
        ----------
        cast_function : function
            Function that converts a coordinate to the new type
        Returns
        -------
        V
            New vector
        """
        new_vec = []
        for coord in self.coords:
            new_vec.append(cast_function(coord))
        return V(new_vec)

    def __add__(self, vec):
        ''' Sum of vectors'''
        if type(vec) is type(self):
            if len(self.coords) == len(vec.coords):
                return V([c2 + c1 for c1, c2 in zip(self.coords, vec.coords)])
            else:
                raise WrongDimensionError(('Cannot add vectors '
                                           'with dimensions {} and '
                                           '{}'.format(len(self.coords),
                                                       len(vec.coords))))
        else:
            try:
                return self + V(vec)
            except:
                message = ('Cannot add \'{}\' '
                           'object to vector.'.format(type(vec).__name__))
                raise TypeError(message)

    def __neg__(self):
        return V([-c for c in self.coords])

    def __sub__(self, vec):
        ''' Subtraction of vectors'''
        return self + (-vec)

    def __mul__(self, k):
        if type(k) == V:
            return V([c1*c2 for c1, c2 in zip(self.coords, k.coords)])
        else:
            return V([c*k for c in self.coords])

    def __str__(self):
        ''' String representation '''
        str_ = "Vector("
        for c in self.coords:
            str_ += str(c) + ','
        str_ += ")"
        return str_

    def __iter__(self):
        for c in self.coords:
            yield c

    def __repr__(self):
        ''' String representation '''
        return self.__str__()

    def size(self):
        ''' Size of vector '''
        return sqrt(sum([c**2 for c in self.coords]))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return self.coords[i]


def inter(v1, v2, t):
    ''' Interpolate two vectors. (0 < t < 1)'''
    if type(v1) is not V:
        v1 = V(v1)
    if type(v2) is not V:
        v2 = V(v2)
    d = v2 - v1
    return v1 + d*float(t)


def dot_prod(v1, v2):
    if type(v1) is not V:
        v1 = V(v1)
    if type(v2) is not V:
        v2 = V(v2)
    sum_ = 0
    for i in range(len(v1)):
        sum_ += v1[i]*v2[i]
    return sum_
    # return sum([c1*c2 for c1, c2 in zip(v1, v2)])


def cross_prod(u, v):
    return V(u[1]*v[2] - u[2]*v[1],
             u[2]*v[0] - u[0]*v[2],
             u[0]*v[1] - u[1]*v[0])


def closest_vector(vector, vectors):
    ''' Find the closest vector in a list and returns it'''
    sorted_vecs = sorted(vectors, key=lambda p: (p - vector).size())
    closest_vec = sorted_vecs[0]
    return closest_vec


def dist(p1, p2):
    if type(p1) is not V:
        p1 = V(p1)
    if type(p2) is not V:
        p2 = V(p2)
    return (p1 - p2).size()


def triangle_area(tri):
    return cross_prod(tri[1] - tri[0], tri[2] - tri[0]).size()/2


def triangle_center(tri):
    return (tri[0] + tri[1] + tri[2])*(1/3.)


def normalize(v):
    return v*(1/v.size())


def cos_vectors(v1, v2):
    if type(v1) is not V:
        v1 = V(v1)
    if type(v2) is not V:
        v2 = V(v2)
    return dot_prod(v1, v2)/(v1.size()*v2.size())
