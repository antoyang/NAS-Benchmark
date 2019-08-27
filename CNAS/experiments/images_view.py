import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from darts.utils import create_exp_dir

class ViewImage1(object):
    """Horizon"""
    def __init__(self, paths):

        super(ViewImage1, self).__init__()
        """Use latex"""
        from matplotlib import rc
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        rc('text', usetex=True)
        self.paths = paths
        self.store_path = './image_view/' + 'graph-{}-{}'.format('exp', time.strftime('%Y%m%d-%H%M%S'))
        create_exp_dir(self.store_path, scripts_to_save=None)
        self.pdf = PdfPages(self.store_path+'/figure.pdf')

    def view(self):
        imgs = []
        for path in self.paths:
            img = plt.imread(path)
            imgs.append(img)

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,16))
        fig.subplots_adjust(hspace=0.1, wspace=0)
        idx = 0
        for i in range(2):
            for j in range(2):
                axs[i,j].xaxis.set_major_locator(plt.NullLocator())
                axs[i,j].yaxis.set_major_locator(plt.NullLocator())
                axs[i,j].imshow(imgs[idx], cmap='bone')
                axs[i,j].set_xlabel(r'$(\alpha_'+str(idx+1) + ')$', fontsize=22)
                plt.tight_layout()
                idx = idx+1
        # save as a high quality image
        self.pdf.savefig(bbox_inches = 'tight', dpi=600)
        # plt.savefig(bbox_inches = 'tight', format='png', dpi=600)
        # plt.show()

class ViewImage2(object):
    """Vetical"""
    def __init__(self, paths):
        super(ViewImage2, self).__init__()
        """Use latex"""
        from matplotlib import rc
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        rc('text', usetex=True)
        self.paths = paths
        self.store_path = './image_view/' + 'graph-{}-{}'.format('exp', time.strftime('%Y%m%d-%H%M%S'))
        create_exp_dir(self.store_path, scripts_to_save=None)
        self.pdf = PdfPages(self.store_path+'/figure.pdf')

    def view(self):
        imgs = []
        for path in self.paths:
            img = plt.imread(path)
            imgs.append(img)

        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16,8))
        fig.subplots_adjust(hspace=0.1)
        idx = 0
        for i in range(4):
                axs[i].xaxis.set_major_locator(plt.NullLocator())
                axs[i].yaxis.set_major_locator(plt.NullLocator())
                axs[i].imshow(imgs[idx], cmap='bone')
                axs[i].set_xlabel(r'$(\alpha_'+str(idx+1) + ')$', fontsize=16)
                plt.tight_layout()
                idx = idx+1
        # save as a high quality image
        self.pdf.savefig(bbox_inches = 'tight', dpi=600)
        # plt.show()

if __name__ == '__main__':
    root = './cell_visualize_rst/'
    #name = 'normal.png'
    name = 'reduction.png'
    ## ----none
    # paths = [root+'graph-exp-20181025-145953/'+name,
    #          root+'graph-exp-20181025-150002/'+name,
    #          root+'graph-exp-20181025-150007/'+name,
    #          root+'graph-exp-20181025-150024/'+name]

    ## ---non-none
    paths = [root+'graph-exp-20181026-212506/'+name,
             root+'graph-exp-20181026-212533/'+name,
             root+'graph-exp-20181026-212941/'+name,
             root+'graph-exp-20181026-213713/'+name]
    vi = ViewImage1(paths)
    vi.view()





