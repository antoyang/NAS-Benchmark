from collections import namedtuple
import json
import os
import glob

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

DARTS_PRIMITIVES = [
    ('none', {}),
    ('pool', {'k': 3, 'type_': 'max'}),
    ('pool', {'k': 3, 'type_': 'avg'}),
    ('skip', {}),
    ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}),
    ('conv', {'c_mul': [1], 'k': 5, 'dil': 1}),
    ('conv', {'c_mul': [], 'k': 3, 'dil': 2}),
    ('conv', {'c_mul': [], 'k': 5, 'dil': 2}),
]

PR_DARTS_PRIMITIVES_4 = [
    ('pool', {'k': 3, 'type_': 'max'}),
    ('pool', {'k': 3, 'type_': 'avg'}),
    ('skip', {}),
    ('conv', {'c_mul': [], 'k': 3, 'dil': 1}),
]

PR_DARTS_PRIMITIVES_5 = [
    ('none', {}),
    ('pool', {'k': 3, 'type_': 'max'}),
    ('pool', {'k': 3, 'type_': 'avg'}),
    ('skip', {}),
    ('conv', {'c_mul': [], 'k': 3, 'dil': 1}),
]

# testing for worst case...
TEST_PRIMITIVES = [
    ('conv', {'c_mul': [1], 'k': 7, 'dil': 1}),
    ('conv', {'c_mul': [3, 1], 'k': 7, 'dil': 1}),
    ('conv', {'c_mul': [3, 3, 1], 'k': 7, 'dil': 1}),
    ('conv', {'c_mul': [3, 3, 1, 1], 'k': 7, 'dil': 1}),
    ('conv', {'c_mul': [2], 'k': 7, 'dil': 1}),
    ('conv', {'c_mul': [3, 2], 'k': 7, 'dil': 1}),
    ('conv', {'c_mul': [3, 3, 2], 'k': 7, 'dil': 1}),
    ('conv', {'c_mul': [3, 3, 2, 1], 'k': 7, 'dil': 1}),
    ('conv', {'c_mul': [3], 'k': 7, 'dil': 1}),
    ('conv', {'c_mul': [3, 3], 'k': 7, 'dil': 1}),
    ('conv', {'c_mul': [3, 3, 3], 'k': 7, 'dil': 1}),
    ('conv', {'c_mul': [3, 3, 3, 1], 'k': 7, 'dil': 1}),
]


def get_init_ops(type_='darts'):
    if type_.startswith('test_'):
        i = int(type_.split('_')[1])
        return TEST_PRIMITIVES[:i]
    return {
        'darts': DARTS_PRIMITIVES,
        'prdarts4': PR_DARTS_PRIMITIVES_4,
        'prdarts5': PR_DARTS_PRIMITIVES_5,
    }.get(type_.lower())


NASNet = Genotype(
    normal=[
        ('conv', {'c_mul': [1], 'k': 5, 'dil': 1}, 1),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 0),
        ('conv', {'c_mul': [1], 'k': 5, 'dil': 1}, 0),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 0),
        ('pool', {'k': 3, 'type_': 'avg'}, 1),
        ('skip', {}, 0),
        ('pool', {'k': 3, 'type_': 'avg'}, 0),
        ('pool', {'k': 3, 'type_': 'avg'}, 0),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 1),
        ('skip', {}, 1)],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('conv', {'c_mul': [1], 'k': 5, 'dil': 1}, 1),
        ('conv', {'c_mul': [1], 'k': 7, 'dil': 1}, 0),
        ('pool', {'k': 3, 'type_': 'max'}, 1),
        ('conv', {'c_mul': [1], 'k': 7, 'dil': 1}, 0),
        ('pool', {'k': 3, 'type_': 'avg'}, 1),
        ('conv', {'c_mul': [1], 'k': 5, 'dil': 1}, 0),
        ('skip', {}, 3),
        ('pool', {'k': 3, 'type_': 'avg'}, 2),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 2),
        ('pool', {'k': 3, 'type_': 'max'}, 1)],
    reduce_concat=[4, 5, 6])

AmoebaNet = Genotype(
    normal=[
        ('pool', {'k': 3, 'type_': 'avg'}, 0),
        ('pool', {'k': 3, 'type_': 'max'}, 1),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 0),
        ('conv', {'c_mul': [1], 'k': 5, 'dil': 1}, 2),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 0),
        ('pool', {'k': 3, 'type_': 'avg'}, 3),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 1),
        ('skip', {}, 1),
        ('skip', {}, 0),
        ('pool', {'k': 3, 'type_': 'avg'}, 1)],
    normal_concat=[4, 5, 6],
    reduce=[
        ('pool', {'k': 3, 'type_': 'avg'}, 0),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 1),
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('conv', {'c_mul': [1], 'k': 7, 'dil': 1}, 2),
        ('conv', {'c_mul': [1], 'k': 7, 'dil': 1}, 0),
        ('pool', {'k': 3, 'type_': 'avg'}, 1),
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('pool', {'k': 3, 'type_': 'max'}, 1),
        ('conv_7x1_1x7', {}, 0),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 5)],
    reduce_concat=[3, 4, 6])

DARTS_V1 = Genotype(
    normal=[
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 1),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 0),
        ('skip', {}, 0),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 1),
        ('skip', {}, 0),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 1),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 0),
        ('skip', {}, 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('pool', {'k': 3, 'type_': 'max'}, 1),
        ('skip', {}, 2),
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('skip', {}, 2),
        ('skip', {}, 2),
        ('pool', {'k': 3, 'type_': 'avg'}, 0)],
    reduce_concat=[2, 3, 4, 5])

DARTS_V2 = Genotype(
    normal=[
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 0),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 1),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 0),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 1),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 1),
        ('skip', {}, 0),
        ('skip', {}, 0),
        ('conv', {'c_mul': [], 'k': 3, 'dil': 2}, 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('pool', {'k': 3, 'type_': 'max'}, 1),
        ('skip', {}, 2),
        ('pool', {'k': 3, 'type_': 'max'}, 1),
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('skip', {}, 2),
        ('skip', {}, 2),
        ('pool', {'k': 3, 'type_': 'max'}, 1)],
    reduce_concat=[2, 3, 4, 5])

PDARTS = Genotype(
    normal=[
        ('skip', {}, 0),
        ('conv', {'c_mul': [], 'k': 3, 'dil': 2}, 1),
        ('skip', {}, 0),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 1),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 1),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 3),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 0),
        ('conv', {'c_mul': [], 'k': 5, 'dil': 2}, 4)],
    normal_concat=list(range(2, 6)),
    reduce=[
        ('pool', {'k': 3, 'type_': 'avg'}, 0),
        ('conv', {'c_mul': [1], 'k': 5, 'dil': 1}, 1),
        ('conv', {'c_mul': [1], 'k': 3, 'dil': 1}, 0),
        ('conv', {'c_mul': [], 'k': 5, 'dil': 2}, 2),
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('conv', {'c_mul': [], 'k': 3, 'dil': 2}, 1),
        ('conv', {'c_mul': [], 'k': 3, 'dil': 2}, 1),
        ('conv', {'c_mul': [], 'k': 5, 'dil': 2}, 3)],
    reduce_concat=list(range(2, 6)))

PR_DARTS_DL1 = Genotype(
    normal=[
        ('conv', {'k': 5, 'dil': 1, 'c_mul': [1]}, 0),
        ('conv', {'k': 3, 'dil': 1, 'c_mul': [1]}, 1),
        ('conv', {'k': 3, 'dil': 1, 'c_mul': [1]}, 1),
        ('skip', {}, 2),
        ('conv', {'k': 3, 'dil': 1, 'c_mul': [1]}, 1),
        ('skip', {}, 2),
        ('skip', {}, 0),
        ('skip', {}, 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('conv', {'k': 3, 'dil': 1, 'c_mul': [1]}, 0),
        ('skip', {}, 1),
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('conv', {'k': 5, 'dil': 1, 'c_mul': [1]}, 2),
        ('conv', {'k': 3, 'dil': 1, 'c_mul': [1]}, 0),
        ('conv', {'k': 3, 'dil': 1, 'c_mul': [1]}, 1),
        ('pool', {'k': 3, 'type_': 'max'}, 1),
        ('conv', {'k': 3, 'dil': 1, 'c_mul': [1]}, 4)],
    reduce_concat=[2, 3, 4, 5])

PR_DARTS_DL2 = Genotype(
    normal=[
        ('conv', {'c_mul': [1], 'dil': 1, 'k': 3}, 0),
        ('conv', {'c_mul': [], 'dil': 1, 'k': 3}, 1),
        ('conv', {'c_mul': [], 'dil': 1, 'k': 3}, 1),
        ('conv', {'c_mul': [1], 'dil': 1, 'k': 7}, 2),
        ('skip', {}, 0),
        ('conv', {'c_mul': [1], 'dil': 1, 'k': 5}, 1),
        ('conv', {'c_mul': [1], 'dil': 1, 'k': 3}, 1),
        ('conv', {'c_mul': [1], 'dil': 1, 'k': 7}, 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('conv', {'c_mul': [], 'dil': 1, 'k': 3}, 0),
        ('conv', {'c_mul': [1], 'dil': 1, 'k': 3}, 1),
        ('conv', {'c_mul': [1], 'dil': 1, 'k': 3}, 1),
        ('skip', {}, 2),
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('conv', {'c_mul': [], 'dil': 1, 'k': 3}, 3),
        ('conv', {'c_mul': [], 'dil': 1, 'k': 3}, 0),
        ('skip', {}, 3)],
    reduce_concat=[2, 3, 4, 5])

PR_DARTS_DR = Genotype(
    normal=[
        ('conv', {'k': 3, 'c_mul': [], 'dil': 1}, 0),
        ('conv', {'k': 3, 'c_mul': [3], 'dil': 1}, 1),
        ('conv', {'k': 7, 'c_mul': [1], 'dil': 1}, 1),
        ('conv', {'k': 5, 'c_mul': [2], 'dil': 1}, 2),
        ('skip', {}, 1),
        ('skip', {}, 3),
        ('skip', {}, 0),
        ('skip', {}, 2)],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('conv', {'k': 5, 'c_mul': [2], 'dil': 1}, 0),
        ('pool', {'k': 3, 'type_': 'max'}, 1),
        ('pool', {'k': 3, 'type_': 'max'}, 1),
        ('conv', {'k': 3, 'c_mul': [1], 'dil': 1}, 2),
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('conv', {'k': 3, 'c_mul': [3], 'dil': 1}, 2),
        ('pool', {'k': 3, 'type_': 'max'}, 0),
        ('conv', {'k': 1, 'c_mul': [1], 'dil': 1}, 3)],
    reduce_concat=[2, 3, 4, 5])

PR_DARTS_UR = Genotype(
    normal=[
        ('conv', {'dil': 1, 'c_mul': [2], 'k': 3}, 0),
        ('conv', {'dil': 1, 'c_mul': [], 'k': 1}, 1),
        ('conv', {'dil': 1, 'c_mul': [1, 1, 1], 'k': 3}, 1),
        ('conv', {'dil': 1, 'c_mul': [], 'k': 5}, 2),
        ('conv', {'dil': 1, 'c_mul': [1], 'k': 1}, 1),
        ('conv', {'dil': 1, 'c_mul': [1, 1], 'k': 3}, 2),
        ('conv', {'dil': 1, 'c_mul': [1], 'k': 1}, 3),
        ('conv', {'dil': 1, 'c_mul': [1, 1], 'k': 5}, 4)],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('conv', {'dil': 1, 'c_mul': [1], 'k': 7}, 0),
        ('conv', {'dil': 1, 'c_mul': [], 'k': 3}, 1),
        ('conv', {'dil': 1, 'c_mul': [1, 1], 'k': 3}, 0),
        ('conv', {'dil': 1, 'c_mul': [1], 'k': 3}, 2),
        ('skip', {}, 0),
        ('skip', {}, 2),
        ('conv', {'dil': 1, 'c_mul': [1, 1], 'k': 7}, 1),
        ('skip', {}, 2)],
    reduce_concat=[2, 3, 4, 5])


def save_genotype(path, genotype):
    with open(path, mode='w+') as file:
        json.dump(genotype, file, ensure_ascii=False, indent=2)


def load_genotype(name_or_path, skip_cons=0):
    try:
        return eval(name_or_path)
    except:
        pass
    for sc in range(skip_cons, -1, -1):
        for maybe_file in glob.glob(name_or_path.replace('%d', str(sc))):
            if os.path.isfile(maybe_file):
                with open(maybe_file, mode='r') as file:
                    data = json.load(file)
                    return Genotype(normal=data[0], normal_concat=data[1], reduce=data[2], reduce_concat=data[3])
    print('could not find file:', name_or_path)
    raise FileNotFoundError
