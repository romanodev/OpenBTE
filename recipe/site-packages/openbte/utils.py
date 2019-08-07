from mpi4py import MPI
import numpy as np



def compute_serial_sum(func,n_tot,output,options):

  for key, value in output.items():
   strc = r'''output[' ''' + key + r''' '].fill(0)'''
   exec("".join(strc).replace(" ", ""))
  #Do the sum
  for n in range(n_tot):
   tmp = func(n,options)
   for key, value in output.items():
    strc = r'''output[' ''' + key + r''' ']+=tmp[' ''' + key +r''' '] '''
    exec(strc.replace(" ", ""))


def compute_parallel_sum(func,n_tot,output,options):

   comm = MPI.COMM_WORLD

   comm.Barrier()

   
   size = comm.Get_size()
   n_left = n_tot
   n_start = 0
   #data = {}
   #VARIABLES------------------------------
   for key, value in output.items():
    strc = r'''output[' ''' + key + r''' '].fill(0)'''
    exec("".join(strc).replace(" ", ""))
    exec(key + '_tot = value')
   #-------------------------------------
   while n_left > 0:
    n_cycle = min([n_left,size-1])
    if comm.Get_rank() > 0:
     n = n_start + comm.Get_rank()-1
     if n < n_tot:
      tmp = func(n,options)
      for key, value in output.items():
       strc = key + r''' = tmp[' ''' + key +r''' '] '''
       exec(strc.replace(" ", ""))
     else:
      for key, value in output.items():
       exec(key + ' = value')
      #----------------------------------

    else :
     for key, value in output.items():
      exec(key + ' = ' + key + '_tot.copy()')

    for key, value in output.items():
     exec('comm.Reduce([' + key+',MPI.DOUBLE],[' + key+'_tot,MPI.DOUBLE],op=MPI.SUM,root=0)')

    n_left  -= n_cycle
    n_start += n_cycle

   comm.Barrier()

   strc = r'''data = {   '''
   for key, value in output.items():
    strc += r''' ' ''' + key + r''' ':  ''' + key +'_tot,'
   strc = list(strc)

   strc[-1] = '}'
   strc = "".join(strc).replace(" ", "")
   
   exec(strc)
   #print(data)
   #print(locals().keys())
   #quit()
   tmp = comm.bcast(data,root=0)
   for key, value in tmp.items():
    if comm.Get_rank() > 0:
     strc = r'''output[' ''' + key+r''' '] +=tmp[' ''' + key+r''' ']'''
     exec("".join(strc).replace(" ", ""))



def compute_sum(func,n_tot,output = {},options = {}):
  if MPI.COMM_WORLD.Get_size() > 1 :
   compute_parallel_sum(func,n_tot,output,options)
  else:
   compute_serial_sum(func,n_tot,output,options)
