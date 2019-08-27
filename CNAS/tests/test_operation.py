import unittest
import time
import torch
import sys
sys.path.append('..')
from darts.operations import *

class TestOperation(unittest.TestCase):
    def setUp(self):
        self.input = torch.randn(40,32,32,32).cuda()
        self.target = torch.randint(0,10, (40,), dtype=torch.long).cuda()
        self.fc = torch.nn.Linear(32, 10).cuda()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1).cuda()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.stride = 2
        self.c = 32
        self.atom = PRIMITIVES

    def classifier(self, layer):
        out = layer(self.input)
        out = self.avgpool(out)
        out = self.fc(out.view(out.size(0), -1))
        loss = self.criterion(out, self.target)
        loss.backward()
        return loss.item()

    def test_op0(self):
        start = time.time()
        layer = OPS[self.atom[0]](self.c, self.stride, False).cuda()
        loss = self.classifier(layer = layer)
        end = time.time()
        print('ops0 none: loss: {0}, cost: {1}s'.format(loss, end-start))

    def test_op1(self):
        start = time.time()
        layer = OPS[self.atom[1]](self.c, self.stride, False).cuda()
        loss = self.classifier(layer = layer)
        end = time.time()
        print('ops1 skip connection: loss: {0}, cost: {1}s'.format(loss, end-start))

    def test_op2(self):
        start = time.time()
        layer = OPS[self.atom[2]](self.c, self.stride, False).cuda()
        loss = self.classifier(layer = layer)
        end = time.time()
        print('ops2 cweight_com: loss: {0}, cost: {1}s'.format(loss, end-start))


    def test_op3(self):
        start = time.time()
        layer = OPS[self.atom[3]](self.c, self.stride, False).cuda()
        loss = self.classifier(layer = layer)
        end = time.time()
        print('ops3 avg_pool_3x3: loss: {0}, cost: {1}s'.format(loss, end-start))

    def test_op4(self):
        start = time.time()
        layer = OPS[self.atom[4]](self.c, self.stride, False).cuda()
        loss = self.classifier(layer = layer)
        end = time.time()
        print('ops4 max_pool_3x3: loss: {0}, cost: {1}s'.format(loss, end-start))

    def test_op5(self):
        start = time.time()
        layer = OPS[self.atom[5]](self.c, self.stride, False).cuda()
        loss = self.classifier(layer = layer)
        end = time.time()
        print('ops5 sep_conv_3x3: loss: {0}, cost: {1}s'.format(loss, end-start))

    def test_op6(self):
        start = time.time()
        layer = OPS[self.atom[6]](self.c, self.stride, False).cuda()
        loss = self.classifier(layer = layer)
        end = time.time()
        print('ops6 dil_conv_3x3: loss: {0}, cost: {1}s'.format(loss, end - start))

    def test_op7(self):
        start = time.time()
        layer = OPS[self.atom[7]](self.c, self.stride, False).cuda()
        loss = self.classifier(layer = layer)
        end = time.time()
        print('ops7 shuffle_conv_3x3: loss: {0}, cost: {1}s'.format(loss, end - start))

if __name__ == "__main__":
    unittest.main()