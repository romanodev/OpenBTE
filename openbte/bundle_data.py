from openbte.utils import *


def main():
 data = {}
 for name in sys.argv[1:]:
  tmp = load_data(name[:-4])
  key = list(tmp.keys())[0]
  data[key] = tmp[key]


 save_data('bundle',data)










