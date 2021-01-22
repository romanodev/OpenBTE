from openbte.utils import *



def read_data(**argv):

    data = {}
    if len(argv) == 0:
     for name in sys.argv[1:]:
       tmp = load_data(name[:-4])
       data[name[:-4]] = tmp
    else:   
     for name,value in argv.items():
       tmp = load_data(name)
       data[name] = value

    return data   




def main():

 
 data = read_data()  


 save_data('bundle',data)










