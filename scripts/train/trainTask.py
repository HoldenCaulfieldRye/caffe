import subprocess
import sys
import os
import getopt
import copy

# python trainModel.py --model=clampdet --submodel=conv4


if __name__ == "__main__":
  opts, extraparams = getopt.gnu_getopt(sys.argv[1:], "", ["model=", "submodel=", "freq_snaps", "train_net=", "test_net=", "test_iter=", "test_interval=", "test_compute_loss=", "base_lr=", "display=", "max_iter=", "lr_policy=", "gamma=", "power=", "momentum=", "weight_decay=", "stepsize=", "snapshot=", "snapshot_prefix=", "snapshot_diff=", "solver_mode=", "device_id=", "random_seed="])
  optDict = dict([(k[2:],v) for (k,v) in opts])
  print optDict
  if not "model" in optDict:
    raise Exception("Need to specify --model flag")
  task = optDict["model"].capitalize()
  if not "submodel" in optDict:
    raise Exception("Need to specify --submodel flag")
  submodel = optDict["submodel"]
  baseDir = os.path.abspath("../models/" + task) + "/"
  logsDir = os.path.abspath(baseDir + "logs/" + submodel) + "/"
  solverFile = baseDir + task + "_solver.prototxt"
  curOpts = [] 
  if "snapshot" in optDict and "freq_snaps" in optDict:
    print "WARNING: \"snapshot\" was specified, ignoring \"freq_snaps\" flag"
    del optDict["freq_snaps"]
  if "freq_snaps" in optDict and "test_interval" in optDict:
    optDict["snapshot"] = optDict["test_interval"] 
    del optDict["freq_snaps"]

  test_interval = None
  with open(solverFile, 'r') as f:
    for line in f:
      opt, val = tuple(map(lambda x: x.strip(), line.strip().split(":")))
      if opt == "test_interval":
        test_interval = val
      if opt in optDict:
        print "Changing", opt, "from", val, "to", optDict[opt]
        val = optDict[opt]
      curOpts.append((opt,val))

  if "snapshot" not in optDict and "freq_snaps" in optDict:
    for i in range(len(curOpts)):
     if curOpts[i][0] == "snapshot":
       print "Changing \"snapshot\" from", curOpts[i][1], "to", test_interval
       curOpts[i] = (curOpts[i][0], test_interval)

  with open (solverFile, 'w') as f:
    for k,v in curOpts:
      f.write(k + ": " + v + "\n")
  
  os.mkdir(logsDir)
  for s in ['_train','_val','_solver']:
    shutil.copy(baseDir + task + s + '.prototxt', logsDir)
  cmd = "cd " + baseDir + "; nohup ./fine_" + task + ".sh 2>&1 | tee " logsDir + "train_output.log &"
  p = subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE, stderr=subprocess.STDOUT)

  lastLine = None 
  newAcc = 0 
  bestSnap = None
  bestAcc = 0
  delete = []
  while True:
    out = p.stdout.readline()
    if out == '' and p.poll() != None:
      break
    if out != '':
      if "Test score #0" in out:
        newAcc = float(out.strip().split()[-1])
      if "Snapshotting to" in out:
        newSnap = out.strip().split()[-1]
        if newAcc > bestAcc:
          if bestSnap:
            delete += [baseDir + bestSnap, baseDir + bestSnap+".solverstate"]
          bestAcc = newAcc
          bestSnap = newSnap
        else:
          delete += [baseDir + newSnap, baseDir + newSnap+".solverstate"]
      if len(delete) > 0 and (not "Snapshotting" in out):
        for f in delete:
          os.remove(f)
        delete = []
      sys.stdout.write(out)
      sys.stdout.flush()
      lastLine = out
 
# val_batch_size = 96
# test_iter = (val_set_size / val_batch_size) + 1








