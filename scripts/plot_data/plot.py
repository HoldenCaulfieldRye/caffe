#!/usr/bin/env python

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join as oj
import subprocess

# Usage: python plot.py path/to/model test-inter=.. [start-iter=..] [end-iter==..]


def matplot(model_dir, Ys, start, end):  
  plt.ylim([0,1.2])
  x = np.array(range(start,end))
  plt.xlabel('Iters')
  for key in Ys.keys():
    Ys[key] = np.array([np.float(el) for el in Ys[key][start:end]])
    plt.plot(x, Ys[key], label=key)
  plt.legend(loc='upper left')
  # plt.title('Go on choose one')
  plt.grid(True)
  plt.savefig(oj(model_dir,'plot_'+model_dir.split('/')[-3]+'_'+model_dir.split('/')[-1]+'.png'))
  # plt.show()

  
def already_parsed(model_dir):
  fnames = []
  listdir = os.listdir(model_dir)
  for fname in listdir:
    if 'train_output' in fname and fname.endswith('.log'):
      fnames.append(oj(model_dir,fname))
  if len(fnames) == 0:
    print "ERROR: no file containing 'train_output' and ending in '.log' found in", model_dir
  elif len(fnames) > 1:
    for elem in enumerate(fnames): print elem
    fname = oj(model_dir,fnames[int(raw_input("\nChoose index number from above: "))])
  else: fname = oj(model_dir,fnames[0])
  if all([os.path.basename(fname)+'.train' in listdir,
          os.path.basename(fname)+'.test' in listdir]):
    return fname, 'Y' # found log and parsed
  else: return fname, 'N' # found log but not parsed
 
  
  
def parse_log(log_name):
  cmd = "./parse_log.sh "+log_name
  p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  p.wait()


def get_train_dict(ltfname):
  ''' Returns a dict of all time series columns in 
  *train_output*.log.train in model dir. '''
  if not os.path.isfile(ltfname):
    raise Exception("ERROR: %s does not exist:"%(ltfname))
  train_dict = columns_to_dict(ltfname)
  return train_dict


def get_test_dict(ltfname):
  ''' Returns a dict of all time series columns in 
  *train_output*.log.test in model dir. Time series stretched by 
  test interval. '''
  if not os.path.isfile(ltfname):
    raise Exception("ERROR: %s does not exist:"%(ltfname))
  test_dict = columns_to_dict(ltfname)
  test_interval = get_test_interval(ltfname)
  test_dict = stretch_time_series(test_dict, test_interval)
  return test_dict

  
def columns_to_dict(fname):
  d = {}
  content = open(fname,'r').readlines()
  stat_names = content[0].split()
  for i in range(len(content)):
    if len(content[i].split()) < len(content[0].split()):
      # if all([content[0] == 'LearningRate',
      #        len(content[0].split())-len(content[i].split()) == 1]):
      #   break
      # else:
      raise Exception('ERROR: line[%i] does not have data for each %s entries.'%(i,stat_names))
  for (idx,name) in enumerate(stat_names):
    d[name] = [elem.split()[idx] for elem in content[1:]]
  return d

    
def get_test_interval(ltfname):
  test = open(ltfname,'r').readlines()
  return int(test[2].split()[0])
  # return len(open(oj(model_dir,'train_output.log.train'),'r').readlines()) / len(open(oj(model_dir,'train_output.log.test'),'r').readlines()) + 1


def stretch_time_series(test_dict, test_interval):
  for key in test_dict.keys():
    stretch = []
    for datp in test_dict[key][:-1]:
      for k in range(test_interval):
        stretch.append(datp)
    stretch.append(test_dict[key][-1])
    test_dict[key] = stretch
  return test_dict


def print_help():
  print '''Usage eg:
  ./plot.py ../../task/scrape/conv1'''

if __name__ == '__main__':

  test_fields = ['TestLoss', 'Acc_0', 'Acc_1', 'PCAcc', 'Accuracy']
  train_fields = ['TrainLoss']
  
  try: 
    os.environ['DISPLAY']
  except: 
    raise Exception('ERROR: X11 forwarding not enabled, cannot run script')

  if len(sys.argv) < 2:
      print_help()
  else:
    
    model_dir = os.path.abspath(sys.argv[1])

    log_fname, parsed = already_parsed(model_dir)
    if parsed == 'N':
      parse_log(log_fname)
    lfname = oj(model_dir, log_fname)

    test_dict = get_test_dict(lfname+'.test')
    train_dict = get_train_dict(lfname+'.train')

    Ys = {}
    tr_te_fields = train_fields + test_fields
    for key in train_fields:
      Ys[key] = train_dict[key]
    for key in test_fields:
      Ys[key] = test_dict[key]

    # assert all time series same length
    keys = Ys.keys()
    for i in range(1,len(tr_te_fields)):
      if len(Ys[tr_te_fields[i]]) != len(Ys[tr_te_fields[0]]):
        raise Exception('%i %s entries vs %i %s entries'%(len(Ys[tr_te_fields[i]]),tr_te_fields[i],len(Ys[tr_te_fields[0]]),tr_te_fields[0]))
             
    start, end = 0, len(Ys['TrainLoss'])
    for arg in sys.argv:
      if arg.startswith("start-iter="):
        start = int(arg.split('=')[-1])
      if arg.startswith("end-iter="):
        end = int(arg.split('=')[-1])
    
    matplot(model_dir, Ys, start, end)

        
