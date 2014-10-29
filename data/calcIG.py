import math
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

def norm(ps):
  n = sum(ps)
  return [float(p)/n for p in ps]

def getH(ps):
  h = 0
  for p in ps:
    h -= p*math.log(p)
  return h

def getIG(ls, bs, verbose=True):
  nls = norm([ls[0] + bs[0], ls[1] + bs[1]])
  h = getH(nls)
  nbs = norm([ls[0] + ls[1], bs[0]+bs[1]])
  lbs = {}
  lbs[(0,0)], lbs[(0,1)], lbs[(1,0)], lbs[(1,1)] = norm([ls[0],bs[0],ls[1],bs[1]])
  ch = 0
  for i in range(2):
    pBlur = nbs[i]
    pConds = [lbs[(0,i)] / pBlur, lbs[(1,i)] / pBlur]
    assert(abs(sum(pConds) - 1.0) < 0.00001)
    hp = getH(pConds) 
    if verbose:
      print "Blur:", i, "with prob", pBlur, "and p(label|blur) =", pConds, "and entropy(p(label|blur)) =", hp
    ch += pBlur * hp

  if verbose:
    print "Entropy:", nls, h
    print "Conditional Entropy:", nbs, lbs, ch
    print "Information Gain:", h - ch
    print "Proportion information gain:", (h-ch) / h
  return (h - ch) / h

def plotIG(bluebox, xs, ys):
  Xs2, Ys2 = np.meshgrid(xs, ys)
  print len(Xs2), len(Ys2)
  Zs2 = []
  for i in range(len(Xs2)):
    Zs2.append([])
    for j in range(len(Ys2)):
      Zs2[i].append(getIG(bluebox, [Xs2[i][j], Ys2[i][j]], verbose=False))

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot_surface(Xs2, Ys2, Zs2, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
  plt.xlabel("Redbox pos", size=20)
  plt.ylabel("Redbox neg", size=20)
  ax.set_zlabel("Information Gain", size=20)
  #plt.title("Trace likelihood when ignoring trans-dimensionality", size=20)
  plt.show()

def getAll(bluebox, redbox):
  return getIG(bluebox, redbox, verbose=False), float(bluebox[0] + redbox[0])/(bluebox[1] + redbox[1]), sum(bluebox) + sum(redbox)

def readLog(fn):
  bluebox = [0,0]
  redbox = [0,0]
  with open(fn, 'r') as f:
    for line in f:
      ind = int(line.strip().split()[-1])
      if "Bluebox" in line:
        bluebox[ind] += 1
      elif "Redbox" in line:
        redbox[ind] += 1
      else:
        raise Exception("No Redbox or Bluebox in Line")
  print "Processed log:", bluebox, redbox
  return getAll(bluebox, redbox)

if __name__ == "__main__":
  print readLog(sys.argv[1])
