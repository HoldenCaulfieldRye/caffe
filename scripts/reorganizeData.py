import sys
import os
import shutil

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
    shutil.rmtree(cdr) 
 
if __name__ == "__main__":
  reorganize(sys.argv[1])
