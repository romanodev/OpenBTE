
from matplotlib.pylab import *

import os,sys


LARGE = 30
c1 = '#1f77b4'
c2 = '#f77f0e'
c3 = '#2ca02c'


def savefigure(prefix = './'):


 namefile =  prefix + sys.argv[0].split('.')[0]+'.png'

 savefig(namefile,dpi=300)




def init_plotting(extra_x_padding = 0.0,extra_y_padding=0.0,extra_bottom_padding = 0.0,extra_top_padding=  0.0,paraview=False,square=True,delta_square = 0,presentation=False):

 rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
 rc('font',**{'family':'serif','sans-serif':['Computer Modern Sans serif']})

 rcParams['xtick.major.pad']='10'
 rc('text', usetex=True)

 rcParams['lines.linewidth'] = 4
 rcParams['font.size'] = LARGE
 rcParams['xtick.labelsize'] = LARGE
 rcParams['ytick.labelsize'] = LARGE
 if square:
  mpl.rcParams['figure.figsize'] = [4.0, 4.0]
 else:
  mpl.rcParams['figure.figsize'] = [8.0, 5.0]
 if presentation:
  mpl.rcParams['figure.figsize'] = [16.0, 12.0]

 rcParams['text.latex.preamble'] = [r'\usepackage{hyperref}',]


 mpl.rcParams['figure.dpi'] = 80
 mpl.rcParams['savefig.dpi'] = 300
 rcParams['xtick.major.size'] = 3
 rcParams['xtick.minor.size'] = 3
 rcParams['figure.edgecolor'] = 'k'
 rcParams['figure.facecolor'] = 'w'
 rcParams['xtick.major.width'] = 1
 rcParams['xtick.minor.width'] = 1
 rcParams['ytick.major.size'] = 3
 rcParams['ytick.minor.size'] = 3
 rcParams['ytick.major.width'] = 1
 rcParams['ytick.minor.width'] = 1
 rcParams['legend.frameon'] = False
 rcParams['legend.fontsize'] = 25
 rcParams['axes.linewidth'] = 1
 #axes([0.15+extra_x_padding,0.20+extra_bottom_padding,0.78-extra_x_padding-extra_y_padding,0.75-extra_bottom_padding-extra_top_padding])
 axes([0,0,1.0,1.0])
