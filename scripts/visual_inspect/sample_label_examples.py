import os
from os.path import join as ojoin
import shutil
import json, yaml, random


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
  json.dump(d, open('label_dict_'+str(date.today()),'w'))
  return d

if __name__ == '__main__':
  here = os.getcwd()
  data_dir = '/data/ad6813/pipe-data/Bluebox/raw_data/dump'
  os.chdir('../scripts/data_preparation')
  d = get_label_dict(data_dir)
  os.chdir(here)
  for label in d.keys():
    if type(d[label]) == list:
      if not os.path.isdir(label): os.mkdir(label)
      length = min(20,len(d[label]))
      for f in d[label][:length]:
        shutil.copy(ojoin(data_dir,f),ojoin(ojoin(label,f)))
