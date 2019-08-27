import unittest
import torch
import torch.nn.functional as F
from darts.model_search import *

class TestModelSearch(unittest.TestCase):
    def setUp(self):
        self.total_input = 14
        self.total_operations = 8
        self.input = torch.randn(8, 32, 32, 32)
        alpha = torch.randn(self.total_input, self.total_operations)
        self.weights = F.softmax(alpha, dim=-1)
        self.num_meta_node = 4
        self.multiplier = 4
        self.inp_c = 3
        self.c = 16
        self.num_classes = 10
        self.layers = 4
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_inp_node = 2

    def test_mixedop(self):
        mop = MixedOp(32, 1)
        rst = mop(self.input, self.weights[0])
        print(rst)

    def test_architecture(self):
        model = Network(self.inp_c, self.c, self.num_classes, self.layers,
                        self.criterion, self.num_inp_node).cuda()
        genotype = model.genotype()

        print('genotype = %s', genotype)
        mini_batch_imgs = torch.randn(8, 3, 32, 32).cuda()
        logits = model(mini_batch_imgs)
        print(logits)



