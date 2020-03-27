from __future__ import print_function
import numpy as np
from scipy import interpolate
import subprocess


def generate_interface(**argv):


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







