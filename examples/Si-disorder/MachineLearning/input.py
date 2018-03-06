from scalify.disorder_interface import *
from scalify.utils_disorder import *
import numpy as np
from scalify.GenerateRandomPoresGrid import *
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

options = {'Lx':40,\
           'Ly':40,\
           'nx':20,\
           'ny':20,\
           'Np':100}
N = 5

fps = 2
duration = N/fps
def make_frame_mpl(n):
 n_iter = float(N*n/duration)
 positions = generate_first_guess(options)
 #positions = np.load('positions')
 fig = plot_sample(options,positions)
 ratio = compute_cost(options,positions)
 np.array(positions).dump(open('positions','wb'))
 title('Iter: ' + str(int(n_iter+1)) + ', S/N: ' + str(round(1/ratio,3)))
 return mplfig_to_npimage(fig) 
 

animation = mpy.VideoClip(make_frame_mpl, duration=duration)
animation.write_gif("simple.gif", fps=fps)


