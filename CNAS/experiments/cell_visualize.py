import sys
import os
import platform
import time
from darts import visualize
from darts.utils import create_exp_dir


def main():

    genotype_name = 'DARTS'
    if len(sys.argv) != 2:
        print('usage:\n python {} ARCH_NAME, Default: DARTS'.format(sys.argv[0]))
    else:
        genotype_name = sys.argv[1]

    store_path = './cell_visualize_pdf/' + 'graph-{}-{}'.format('exp', time.strftime('%Y%m%d-%H%M'))
    create_exp_dir(store_path, scripts_to_save=None)

    if 'Windows' in platform.platform():
        os.environ['PATH'] += os.pathsep + '../3rd_tools/graphviz-2.38/bin/'
    try:
        genotype = eval('geno_types.{}'.format(genotype_name))
    except AttributeError:
        print('{} is not specified in geno_types.py'.format(genotype_name))
        sys.exit(1)

    visualize.plot(genotype.normal, store_path+'/normal')
    visualize.plot(genotype.reduce, store_path+'/reduction')

if __name__ == '__main__':
    main()
