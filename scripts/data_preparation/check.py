import os,sys

# usage: python check.py data-dir=../../data/? data-info=../../data_info/?

def check(data_dir, data_info):
  os.chdir(data_dir)
  d = os.listdir('test')
  
  os.chdir(data_info)
  di = open('test.txt','r').readlines()
  di = [l.split()[0] for l in di]

  if len(d) != len(di):
    print 'not even same number of files'

  count = 0
  for f in d:
    if f not in di: count += 1

  count_2 = 0
  for f in di:
    if f not in d: count_2 += 1

  return [count,count_2]


if __name__ == '__main__':

  for arg in sys.argv:
    if "data-dir=" in arg:
      data_dir = os.path.abspath(arg.split('=')[-1])
    elif "data-info=" in arg:
      data_info = os.path.abspath(arg.split('=')[-1])

  count, count_2 = check(data_dir, data_info)

  print '%i files in data/test not mentioned in data_info/test.txt'%(count)
  print '%i files mentioned in data_info/test.txt not present in data/test'%(count_2)

