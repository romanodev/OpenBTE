from openbte.objects import SolverResults,Mesh,Material
import numpy as np

def HeatSource(mesh     : Mesh,\
               mapping : dict = {})-> SolverResults:
    """A Solver for Heat Source"""

    H = np.zeros(mesh.n_elems) 
    for key,value in mapping.items():
           H[mesh.elem_physical_regions[key]] += value

    variables = {'Heat_Source':{'data':H,'units':'W/m^d'}}


    return SolverResults(variables=variables)
