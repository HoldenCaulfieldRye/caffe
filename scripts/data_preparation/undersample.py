#!/usr/bin/env python
import sys

# delete_some_files('/data/ad6813/caffe/data/scrape/train_2.txt','0',5400)
def delete_some_files(fname, class_num, del_num):
  count, new = 0, []
  content = open(fname, 'r').readlines()
  for line in content:
    if line.endswith(class_num+'\n') and count < del_num: count += 1
    else: new.append(line)
  open(fname,'w').writelines(new)

def print_help():
  print "Usage: ./undersample.py path/to/file class_num del_num"
  print "Example: ./undersample.py /data/ad6813/caffe/data/scrape/train_2.txt 0 5400"

if __name__ == '__main__':
  if len(sys.argv) < 4:
    print_help()
    exit
  else:
    delete_some_files(sys.argv[1], sys.argv[2], int(sys.argv[3]))

