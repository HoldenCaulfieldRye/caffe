#!/usr/bin/env python
import sys, os, glob, shutil
from os.path import join as oj

# expected input: ./reorganizeData.py ../../../pipe-data/Bluebox

def reorganize(baseDir):
  print 'reorganising dirs'
  dirs = filter(lambda x: x.isdigit(), os.listdir(baseDir))
  print len(dirs), "joints found"
  tdr = baseDir 
  if not os.path.exists(tdr):
    os.makedirs(tdr)
  for dr in dirs:
    cdr = baseDir + "/" + dr
    jpgs = filter(lambda x: "jpg" in x, os.listdir(cdr))
    for jpg in jpgs:
      name = jpg.split(".")[0]
      shutil.move(cdr+"/"+jpg, tdr+"/"+jpg) 
      shutil.copyfile(cdr+"/inspection.txt", tdr+"/"+name+".dat")
      shutil.copyfile(cdr+"/meta.txt", tdr+"/"+name+".met")
      getUnsuitableFlags(tdr, name)
    shutil.rmtree(cdr) 
 
def getUnsuitableFlags(tdr, name):
  with open(tdr+"/"+name+".met") as met_f:
    if any(["UnsuitablePhoto=True" in line for line in met_f.readlines()]):
      with open(tdr+"/"+name+".dat", 'a') as dat_f:
        dat_f.write("UnsuitablePhoto")

if __name__ == "__main__":
  baseDir = sys.argv[1]
  tdr = os.path.abspath(baseDir)
  print 'checking whether any joint dirs left...'
  if any([os.path.isdir(oj(tdr,fd)) for fd in os.listdir(tdr)]):
    print 'found some; reorganizing them'
    reorganize(tdr)
  print 'no more joint dirs left'
  jpgs = filter(lambda x: "jpg" in x, os.listdir(tdr))
  for jpg in jpgs:
    name = jpg.split(".")[0]
    getUnsuitableFlags(tdr, name)

