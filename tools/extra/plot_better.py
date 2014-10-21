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

  
def parse_log(model_dir):
  cmd = "./parselog.sh "+oj(model_dir,'train_output.log')
  p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  p.wait()


def get_train_dict(model_dir):
  ''' Returns a dict of all time series columns in 
  *train_output*.log.train in model dir. '''
  data_files = []
  for fname in os.listdir(model_dir):
    if 'train_output' in fname and fname.endswith('.log.train'): data_files.append(fname)
  if len(data_files) != 1:
    print "there is not exactly 1 filename otf '*train_output*.log.train' in given directory"
    sys.exit()
  train_dict = columns_to_dict(oj(model_dir, data_files[0]))
  return train_dict


def get_test_dict(model_dir):
  ''' Returns a dict of all time series columns in 
  *train_output*.log.test in model dir. Time series stretched by 
  test interval. '''
  data_files = []
  for fname in os.listdir(model_dir):
    if 'train_output' in fname and fname.endswith('.log.test'): data_files.append(fname)
  if len(data_files) != 1:
    print "there is not exactly 1 filename otf '*train_output*.log.test' in given directory"
    sys.exit()
  test_dict = columns_to_dict(oj(model_dir, data_files[0]))
  test_interval = get_test_interval(data_files[0])
  test_dict = stretch_time_series(test_dict, test_interval)
  return test_dict

  
def columns_to_dict(fname):
  d = {}
  content = open(fname,'r').readlines()
  stat_names = content[0].split()
  for i in range(len(content)):
    if len(content[i]) < len(content[0]):
      print 'ERROR: line[%i] does not have all %s entries.'%(i,stat_names)
      sys.exit()
  for (idx,name) in enumerate(stat_names):
    d[name] = [elem.split()[idx] for elem in content[1:]]
  return d

    
def get_test_interval(model_dir):
  test = open(fname,'r').readlines()
  return int(test[2].split()[0])
  # return len(open(oj(model_dir,'train_output.log.train'),'r').readlines()) / len(open(oj(model_dir,'train_output.log.test'),'r').readlines()) + 1


def stretch_time_series(test_dict, test_interval):
  for key in test_dict.keys():
    stretch = []
    for i in range(len(test_dict[key])-1):
      for k in range(test_interval):
        stretch.append(test_dict[key][i])
    stretch.append(test_dict[key][-1])
    test_dict[key] = stretch
  return test_dict
    

if __name__ == '__main__':
  print('')
  try: 
    os.environ['DISPLAY']
  except: 
    print 'ERROR: X11 forwarding not enabled, cannot run script'
    sys.exit()

  if len(sys.argv) < 2:
      print_help()
  else:
    
    model_dir = os.path.abspath(sys.argv[1])

    parse_log(model_dir)

    test_dict = get_test_dict(model_dir)
    train_dict = get_train_dict(model_dir)

    Ys = {}
    # SELECT WHICH TRAIN DATA based on column heading names
    for key in ['TrainLoss']:
      Ys[key] = train_dict[key]
    # SELECT WHICH TEST DATA based on column heading names
    for key in ['TestLoss', 'Acc_0', 'Acc_1', 'PCAcc', 'Accuracy']:
      Ys[key] = test_dict[key]

    # assert all time series same length
    keys = Ys.keys()
    for i in range(1,len(keys)):    
      assert len(Ys[keys[i]]) == len(Ys[keys[0]])
      
    start, end = 0, len(Ys['TrainLoss'])
    for arg in sys.argv:
      if arg.startswith("start-iter="):
        start = int(arg.split('=')[-1])
      if arg.startswith("end-iter="):
        end = int(arg.split('=')[-1])
    
    matplot(model_dir, Ys, start, end)

        
