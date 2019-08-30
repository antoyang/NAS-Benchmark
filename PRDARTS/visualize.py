import sys
import genotypes
from graphviz import Digraph
from operations import OPS


def plot(genotype, filename):
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2*i, 2*i + 1]:
            op, op_kwargs, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(i)
            g.edge(u, v, label=OPS.get(op).label_str(**op_kwargs), fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Visualize")
    p.add_argument('--genome', type=str, default='PDARTS')  # PDARTS
    args = p.parse_args()

    try:
        genotype = eval('genotypes.{}'.format(args.genome))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(args.genome))
        sys.exit(1)

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")

