from openbte.utils import *


def main():
 save_data('bundle',{name[:-4]:load_data(name[:-4]) for name in sys.argv[1:]})









