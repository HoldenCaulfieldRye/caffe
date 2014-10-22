import os
from os.path import join as ojoin
from operator import itemgetter as ig
import itertools 
from shutil import rmtree

def get_label_dict(data_dir):
  total_num_images = 0
  path = data_dir
  for fname in os.listdir(os.getcwd()):
    if not fname.startswith('label_dict'): continue
    else:
      if raw_input('\nfound %s; use as label_dict? ([Y]/N) '%(fname)) in ['','Y']:
        return yaml.load(open(fname,'r'))
  d = {'Perfect': []}
  print 'generating dict of label:files from %s...'%(data_dir)
  for filename in os.listdir(path):
    if not filename.endswith('.dat'): continue
    total_num_images += 1
    fullname = os.path.join(path, filename)
    with open(fullname) as f:
      content = [line.strip() for line in f.readlines()] 
      if content == []:
        d['Perfect'].append(filename.split('.')[0]+'.jpg')
      else:
        for label in content:
          if label not in d.keys(): d[label] = []
          d[label].append(filename.split('.')[0]+'.jpg')
  d['total_num_images'] = total_num_images
  return d
