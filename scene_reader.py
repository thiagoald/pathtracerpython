from os.path import dirname, join
from vector import V, cross_prod, dist, triangle_area, normalize


def calc_normal(tri):
    normal = normalize(cross_prod(tri[1] - tri[0],
                                  tri[2] - tri[0]))
    return normal


def next_tokens(lines):
    if lines != []:
        line = lines[0]
        # print(line.replace('\n', ''))
        del lines[0]
        tokens = [t for t in line.split(' ') if not t in ['', ' ']]
        return tokens
    else:
        return []


def look_ahead(lines):
    line = lines[0]
    tokens = [t for t in line.split(' ') if not t in ['', ' ']]
    # print('tokens: ', tokens)
    return tokens


def remove_spaces_from_start(line):
    i = 0
    while line[i] == ' ':
        i += 1
    return line[i:]


def remove_all_comments(lines):
    new_lines = []
    for line in lines:
        line = remove_spaces_from_start(line)
        if line[0] != '#':
            if '#' in line:
                new_lines.append(line.split('#')[0])
            else:
                new_lines.append(line)
            new_lines[-1] = new_lines[-1].replace('\n', '').replace('\t', ' ')
    return new_lines


class Obj():
    def __init__(self, path):
        print('Reading ' + path)

        self.triangles = []
        self.areas = []
        self.normals = []
        self.vertexes = []
        self.faces = []
        self.vtx_idx = 0

        self.read_obj(path)

    def parse_vertex(self, tokens):
        self.vertexes.append(V([float(x) for x in tokens]))
        self.vtx_idx += 1

    def parse_face(self, tokens):
        face = []
        for i in tokens:
            i = int(i)
            if i < 0:
                face.append(self.vtx_idx + i)
            else:
                face.append(i - 1)
        faces = []
        if len(face) > 3:
            v1 = face[0]
            for i in range(1, len(face) - 1):
                faces.append((v1, face[i], face[i + 1]))
        else:
            faces.append(face)
        self.faces.extend(faces)
        for face in faces:
            triangle = (self.vertexes[face[0]],
                        self.vertexes[face[1]],
                        self.vertexes[face[2]])
            self.triangles.append(triangle)
            self.normals.append(calc_normal(triangle))
            self.areas.append(triangle_area(triangle))
        # print('f ', self.faces[-1])

    def read_obj(self, path):
        with open(path, 'r') as f:
            self.lines = f.readlines()
            self.lines = remove_all_comments(self.lines)
        while self.lines != []:
            tokens = next_tokens(self.lines)
            if tokens != []:
                if tokens[0] == 'v':
                    self.parse_vertex(tokens[1:])
                elif tokens[0] == 'f':
                    self.parse_face(tokens[1:])
                else:
                    print(f'{path}\n\tSkipping command \'{tokens[0]}\' ! '
                          f'\n\tParameters: {tokens[1:]}')


class Scene():
    def __init__(self, path):
        self.eye = None
        self.width = None
        self.height = None
        self.ortho = None
        self.background = None
        self.ambient = None
        self.light_obj = None
        self.light_color = None
        self.npaths = None
        self.tonemapping = None
        self.seed = None
        self.objects = []
        self.read_scene(path)

    def __repr__(self):
        return (f'<Scene'
                f'\n\t eye = {self.eye}'
                f'\n\t width = {self.width}'
                f'\n\t height = {self.height}'
                f'\n\t ortho = {self.ortho}'
                f'\n\t background = {self.background}'
                f'\n\t ambient = {self.ambient}'
                f'\n\t light_obj = {self.light_obj}'
                f'\n\t light_color = {self.light_color}'
                f'\n\t npaths = {self.npaths}'
                f'\n\t tonemapping = {self.tonemapping}'
                f'\n\t seed = {self.seed}'
                f'\n\t objects = {self.objects}'
                '\n>')

    def read_scene(self, path):
        try:
            with open(path, 'r') as f:
                self.lines = f.readlines()
                self.lines = remove_all_comments(self.lines)
        except OSError as e:
            print('Could not open file!')
            raise e

        while self.lines != []:
            tokens = next_tokens(self.lines)
            if tokens != []:
                if tokens[0] == 'eye':
                    self.eye = [float(tkn) for tkn in tokens[1:4]]
                elif tokens[0] == 'size':
                    self.width = int(tokens[1])
                    self.height = int(tokens[2])
                elif tokens[0] == 'ortho':
                    self.ortho = [float(tkn) for tkn in tokens[1:5]]
                elif tokens[0] == 'background':
                    self.background = [float(tkn) for tkn in tokens[1:4]]
                elif tokens[0] == 'ambient':
                    self.ambient = float(tokens[1])
                elif tokens[0] == 'light':
                    self.light_obj = Obj(join(dirname(path), tokens[1]))
                    self.light_color = [float(tkn) for tkn in tokens[2:6]]
                elif tokens[0] == 'npaths':
                    self.npaths = int(tokens[1])
                elif tokens[0] == 'tonemapping':
                    self.tonemapping = float(tokens[1])
                elif tokens[0] == 'seed':
                    self.seed = int(tokens[1])
                elif tokens[0] == 'object':
                    object_dict = {}
                    object_dict['geometry'] = Obj(
                        join(dirname(path), tokens[1]))
                    object_dict['red'] = float(tokens[2])
                    object_dict['green'] = float(tokens[3])
                    object_dict['blue'] = float(tokens[4])
                    object_dict['ka'] = float(tokens[5])
                    object_dict['kd'] = float(tokens[6])
                    object_dict['ks'] = float(tokens[7])
                    object_dict['kt'] = float(tokens[8])
                    object_dict['n'] = float(tokens[9])
                    self.objects.append(object_dict)
                elif tokens[0] == 'output':
                    self.output = join(dirname(path), tokens[1])
                else:
                    print(f'Scene {path}\n\tSkipping command \'{tokens[0]}\'!'
                          f' Parameters: {tokens[1:]}')
