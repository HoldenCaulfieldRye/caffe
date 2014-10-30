import sys
import os
import shutil
from os.path import join as oj

# expected input: python reorganizeData.py ../../../pipe-data/Bluebox

def reorganize(baseDir):
  dirs = filter(lambda x: x.isdigit(), os.listdir(baseDir))
  print len(dirs), "joints found"
  tdr = baseDir + "/raw_data/dump"
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
  tdr = baseDir + "/raw_data/dump"
  if os.path.isdir(tdr):
    jpgs = filter(lambda x: "jpg" in x, os.listdir(tdr))
    for jpg in jpgs:
      name = jpg.split(".")[0]
      getUnsuitableFlags(tdr, name)
  else: reorganize(baseDir)

