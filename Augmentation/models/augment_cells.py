""" CNN cell for network augmentation """
import torch
import torch.nn as nn
from models import ops
import genotypes as gt


class AugmentCell(nn.Module):
    """ Cell for augmentation
    Each edge is discrete.
    """
    def __init__(self, genotype, i, C_pp, C_p, C, reduction_p, reduction, SSC):
        super().__init__()
        self.reduction = reduction
        try:
            self.n_nodes = len(genotype.normal)
        except:
            self.n_nodes = len(genotype["cell_0"])
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0)

        # generate dag
        if reduction:
            try:
                gene = genotype.reduce
                self.concat = genotype.reduce_concat
            except:
                gene = genotype["cell_%d" % i]
                self.concat = range(2, 2+self.n_nodes)
        else:
            try:
                gene = genotype.normal
                self.concat = genotype.normal_concat
            except:
                gene = genotype["cell_%d" % i]
                self.concat = range(2, 2 + self.n_nodes)
        self.dag = gt.to_dag(C, gene, SSC, reduction)

    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        s_out = torch.cat([states[i] for i in self.concat], dim=1)

        return s_out
