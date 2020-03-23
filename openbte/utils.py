import h5py
from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np


def create_layer(data,f):

  for i in data.keys():
         if not isinstance(i,str): 
                ind = str(i)
         else:  
                ind = i
         if isinstance(data[i],dict):
            g = f.create_group(ind)
            create_layer(data[i],g)
         else:
             f.create_dataset(ind,data=data[i],compression="gzip",compression_opts=9) 


def save_dictionary(data,filename):

 with h5py.File(filename, "w") as f:
      create_layer(data,f)


def load_layer(f):

     tmp = {}
     for i in f.keys():
        if i.isdigit():
               ind = int(i)
        else:   
               ind = i 
        if isinstance(f[i],h5py._hl.group.Group):
            name = f[i].name.split('/')[-1]
            if name.isdigit():
               ind2 = int(name)
            else:   
               ind2 = name 

            tmp.update({ind2:load_layer(f[i])}) 
        else:
            tmp.update({ind:np.array(f[i])}) 
     return tmp 




def load_dictionary(filename):

 data = {}
 with h5py.File(filename, "r") as f:
     data.update({'root':load_layer(f)})


 return data['root']



def download_file(file_id,filename):
      gdd.download_file_from_google_drive(file_id=file_id,
                                           dest_path='./' + filename,showsize=True,overwrite=True)
