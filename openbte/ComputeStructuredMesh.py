

def compute_structured_mesh(**argv):


    lx = argv['lx']
    ly = argv['ly']
    nx = argv['nx']
    ny = argv['ny']
    grid = argv['grid']

    step_x = lx/nx
    step_y = lx/ny

    store = open('mesh.geo', 'w+')




    #store.write('h='+str(mesh_ext) + ';\n')
   
    

    store.close()
    
