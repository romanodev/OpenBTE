import matplotlib.font_manager as fm
from matplotlib.pylab import *

import os,sys


LARGE = 26
c1 = '#1f77b4'
c2 = '#f77f0e'
c3 = '#2ca02c'


def savefigure(prefix = './'):


 namefile =  prefix + sys.argv[0].split('.')[0]+'.png'

 savefig(namefile,dpi=300)


def finalize_plotting(fonts):

 
       for child in gca().get_children(): 
        x = isinstance(child, matplotlib.text.Text)
        if x:

         child.set_font_propertie = fonts['regular']

       for child in gcf().get_children(): 
        x = isinstance(child, matplotlib.text.Text)
        if x:
         
         child.set_font_properties = fonts['regular']

       for label in gca().get_xticklabels():
        label.set_fontproperties(fonts['regular'])

       for label in gca().get_yticklabels():
        label.set_fontproperties(fonts['regular'])
    #-------------------





def init_plotting(extra_x_padding = 0.0,extra_y_padding=0.0,extra_bottom_padding = 0.0,extra_top_padding=  0.0,paraview=False,square=False,delta_square = 0,presentation=False):

 #rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
 #rc('font',**{'family':'serif','sans-serif':['Computer Modern Sans serif']})

 #rcParams['font.family'] = 'serif'
 #rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
 #rcParams['font.serif'] = 'Computer Modern Sans Roman'


 rcParams['xtick.major.pad']='10'
 #rc('text', usetex=True)

 rcParams['mathtext.fontset'] = 'cm'

 rcParams['lines.linewidth'] = 4
 rcParams['font.size'] = LARGE
 rcParams['xtick.labelsize'] = LARGE
 rcParams['ytick.labelsize'] = LARGE
 if square:
  rcParams['figure.figsize'] = [4.0, 4.0]
 else:
  rcParams['figure.figsize'] = [8.0, 5.0]
 if presentation:
  rcParams['figure.figsize'] = [16.0, 12.0]

 #print(rcParams.keys())
 #quit()
 
 #rcParams['text.latex.preamble'] = [r'\usepackage{hyperref}',]

 #load fonts----
 filename = os.path.dirname(__file__) + '/fonts/cmunrm.ttf'
 prop_regular = fm.FontProperties(fname=filename,family='serif')
 filename = os.path.dirname(__file__) + '/fonts/cmunbx.ttf'
 prop_bold = fm.FontProperties(fname=filename,family='serif')
 filename = os.path.dirname(__file__) + '/fonts/cmunti.ttf'
 prop_italic = fm.FontProperties(fname=filename,family='serif')
 fonts = {'regular':prop_regular,'italic':prop_italic,'bold':prop_bold}

 #-----------------------------------

 rcParams['figure.dpi'] = 80
 rcParams['savefig.dpi'] = 300
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
 axes([0.15+extra_x_padding,0.20+extra_bottom_padding,0.78-extra_x_padding-extra_y_padding,0.75-extra_bottom_padding-extra_top_padding])
 #axes([0,0,1.0,1.0])i
 return fonts
