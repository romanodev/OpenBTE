import subprocess
import numpy as np
import scipy.interpolate as interpolate


def periodic_kernel(x1, x2, p,l,variance):
    return variance*np.exp(-2/l/l * np.sin(np.pi*abs(x1-x2)/p) ** 2)

def gram_matrix(xs,p,l,variance):
    return [[periodic_kernel(x1,x2,p,l,variance) for x2 in xs] for x1 in xs]


def generate_random_interface(p,l,variance,scale):

 xs = np.arange(-p/2, p/2,p/200)
 mean = [0 for x in xs]
 gram = gram_matrix(xs,p,l,variance)
 ys = np.random.multivariate_normal(mean, gram)*scale
 f = interpolate.interp1d(xs, ys,fill_value='extrapolate')

 return f




def CreateCorrelation(**argv):


    p = argv['lx']
    l = argv.setdefault('l',1)
    scale = argv.setdefault('scale',1)
    variance = argv.setdefault('variance',1)
    f = generate_random_interface(p,l,variance,scale)

    elem_mat_map = {}
    for ne,c in enumerate(argv['centroids']):
      dd = f(c[1])
      if c[0] < dd:
       elem_mat_map.update({ne:0})
      else:
       elem_mat_map.update({ne:1})

    return elem_mat_map
   







def GenerateInterface(**argv):

    s = r'''    

    Point(1) = {- delta /2, -D, 0, h1 };
    Point(2) = { delta /2, -D, 0, h1 };
    Point(3) = { delta /2, D, 0, h1 } ;
    Point(4) = { -delta /2, D, 0, h1 };
    Point(5) = {- D , - D , 0, h2 };
    Point(6) = { D , -D, 0, h2 };
    Point(7) = { D , D , 0, h2 };
    Point(8) = { -D , D , 0, h2 };

    // Lines
    Line(1) = {1, 2};
    Line(2) = {2, 3};
    Line(3) = {3, 4};
    Line(4) = {4, 1};
    Line(5) = {5, 1};
    Line(6) = {6, 2};
    Line(7) = {7, 6};
    Line(8) = {3, 7};
    Line(9) = {8, 4};
    Line(10) = {8, 5};

    // Surfaces
    Line Loop(1) = {4, -5, -10, 9};
    Plane Surface(1) = {1};
    Line Loop(2) = {8, 7, 6, 2};
    Plane Surface(2) = {2};
    Line Loop(3) = {1, 2, 3, 4};
    Plane Surface(3) = {3};

    // Meshing
    Transfinite Line {4} = Ny Using Progression 1;
    Transfinite Line {2} = Ny Using Progression 1;
    Transfinite Line {1} = Nx Using Progression 1;
    Transfinite Line {3} = Nx Using Progression 1;
    Transfinite Surface {3};
    Recombine Surface {3};
    
    Physical Surface('Matrix') = {1,2,3};

    Physical Line('Periodic_1') = {5,1,6};
    Physical Line('Periodic_2') = {9,3,8};
    Physical Line('Periodic_3') = {10};
    Physical Line('Periodic_4') = {7};

    Periodic Line{5}={9};
    Periodic Line{6}={-8};
    Periodic Line{1}={-3};
    Periodic Line{10}={7};

    '''

    s = s.replace('h1','10')
    s = s.replace('h2','3')
    s = s.replace('Nx','20')
    s = s.replace('Ny','100')
    s = s.replace('D',str(argv['lx']))
    s = s.replace('delta','2')

    f = open('mesh.geo','w+')

    f.write(s)

    f.close()
    subprocess.check_output(['gmsh','-optimize_netgen','-format','msh2','-2','mesh.geo','-o','mesh.msh'])









