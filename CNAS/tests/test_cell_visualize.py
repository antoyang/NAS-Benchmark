import unittest
import os
import platform
from visualize import plot

class TestCellVis(unittest.TestCase):

    def setUp(self):
        self.cell_name = 'DARTS'
        if 'Windows' in platform.platform():
            os.environ['PATH'] += os.pathsep + '../3rd_tools/graphviz-2.38/bin/'

    def test_plot(self):
        genotype = eval('geno_types.{}'.format(self.cell_name))
        plot(genotype.normal, 'normal')
        plot(genotype.reduce, 'reduction')
