import unittest
import torch
import matplotlib.pyplot as plt

from darts.utils import *

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.input = torch.randn(4,3,32,32)
        self.img = torch.randn(3, 32, 32)

    def test_cutout(self):
        self.fig = plt.figure()
        self.left = self.fig.add_subplot(1, 2, 1)
        self.left.set_title('Input Image')
        plt.imshow(self.img.numpy().transpose(1,2,0))
        self.right = self.fig.add_subplot(1,2,2)
        len = 10
        cutout = Cutout(len)
        print('original img size: ', self.img.size())
        cut_img = cutout(self.img)
        print('after cutout img size: ', cut_img.size())

        self.right.set_title('Output Image')
        plt.imshow(cut_img.numpy().transpose(1,2,0))
        plt.show()


    def test_get_freer_gpu(self):
        print(get_freer_gpu())

    def test_calc_parameters_count(self):
        test_conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        test_conv(self.input)

        print('param counts: ', calc_parameters_count(test_conv), ' MB')


if __name__ == '__main__':
    unittest.main()


