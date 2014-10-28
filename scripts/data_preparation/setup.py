#!/usr/bin/env python
import numpy as np
import os
from os.path import join as oj
from PIL import Image
from operator import itemgetter as ig
import itertools 
from datetime import date
from shutil import rmtree
import random, subprocess
import add_redboxes as ar

# expected input: ./setup.py --task= --box= --learn= [--target-bad-min=]

# ./setup.py --task=clamp --box=blue --learn=3-10-14

def main(data_dir, data_info, task, pos_class,target_bad_min=None):
  ''' This is the master function. data_dir: where raw data is. data_info: where to store .txt files. '''
  Keep = get_label_dict_knowing(data_dir, task, pos_class)
  if target_bad_min is not None:
    print "target bad min: %s" %(target_bad_min)
    Keep = rebalance(Keep, total_num_images, target_bad_min)
  Keep = within_class_shuffle(Keep)
  print 'finished shuffling'
  dump_to_files(Keep, data_info, task, data_dir)


def get_label_dict_knowing(data_dir, task, pos_class):
  ''' get_label_dict() knowing exactly which flags to look for and 
  how to group them into classes. 
  task is the name of what we're learning to detect,
  pos_class is a list of the actual flag names to look for. '''
  d = {'Default': [], task: []}
  print 'generating specific dict of class:files from %s...'%(data_dir)
  for filename in os.listdir(data_dir):
    if not filename.endswith('.dat'): continue
    with open(oj(data_dir, filename)) as f:
      content = [line.strip() for line in f.readlines()]
      if any([label==line for (label,line)
              in itertools.product(pos_class,content)]):
        d[task].append(filename.split('.')[0]+'.jpg')
      else:
        d['Default'].append(filename.split('.')[0]+'.jpg')
  return d


# def classes_to_learn(lab_to_learn):
#   classes = lab_to_learn.split(' ')
#   Keep = {}
#   print ''
#   for elem in enumerate(sorted(All.keys())): print elem
#   read_labels = [sorted(All.keys())[int(num)] for num in raw_input("\nNumbers of labels to learn, separated by ' ': ").split()]
#   # if 'Perfect' in All.keys():
#   #   Keep['Perfect'] = All['Perfect']
#   for label in read_labels:
#     Keep[label] = All[label]
#   return Keep


def rebalance(Keep, total_num_images, target_bad_min):
  '''if target_bad_min not given, prompts user for one; 
  and implements it. Note that with >2 classes, this can be 
  implemented either by downsizing all non-minority classes by the
  same factor in order to maintain their relative proportions, or 
  by downsizing as few majority classes as possible until
  target_bad_min achieved. We can assume that we care mostly about 
  having as few small classes as possible, so the latter is 
  implemented.'''
  if target_bad_min == 'N': return Keep
  else: target_bad_min = float(target_bad_min)
  # minc is class with minimum number of training cases
  ascending_classes = sorted([(key,len(Keep[key]))
                              for key in Keep.keys()],
                             key=lambda x:x[1])
  maxc, len_maxc = ascending_classes[-1][0], ascending_classes[-1][1]
  minc, len_minc = ascending_classes[0][0], ascending_classes[0][1]
  # print ascending_classes
  # print "\ntotal num images: %i"%(total_num_images)
  maxc_proportion = float(len_maxc)/total_num_images
  if target_bad_min is None:
    target_bad_min = raw_input("\nmax class currently takes up %.2f, what's your target? [num/N] "%(maxc_proportion))
  if target_bad_min is not 'N':
    target_bad_min = float(target_bad_min)
    print 'maxc_proportion: %.2f, target_bad_min: %.2f'%(maxc_proportion, target_bad_min)
    if maxc_proportion > target_bad_min:
      delete_size = int((len_maxc - (target_bad_min*total_num_images))/(1-target_bad_min))
      random.shuffle(Keep[maxc])
      print '%s has %i images so %i will be randomly removed'%(maxc, len_maxc, delete_size)
      del Keep[maxc][:delete_size]
    elif maxc_proportion < target_bad_min:
      print 'woah, you want to INCREASE class imbalance!'
      delete_size = int(total_num_images - (len_maxc/float(target_bad_min)))
      random.shuffle(Keep[minc])
      print '%s has %i images so %i will be randomly removed'%(minc, len_minc, delete_size)
      del Keep[minc][:delete_size]
  assert target_bad_min == round(float(len(Keep[maxc])) / (len(Keep[maxc])+len(Keep[minc])), 2)
  return Keep


def default_class(All, Keep):
  ''' all images without retained labels go to default class. '''
  label_default = "Default" #raw_input("\nDefault label for all images not containing any of given labels? (name/N) ")
  if label_default is not 'N':
    Keep[label_default] = All['Perfect']
    # no need to check for overlap between Perfect and Keep's other
    # labels because Perfect overlaps with no other label by def
    # below is why need to wait for merge_classes
    for key in All.keys():
      if key in Keep.keys()+['Perfect']: continue
      else:
        # computationally inefficient. but so much more flexible to
        # have this dict.
        # add fname if not in any
        # ---
        # updating 'already' is the expensive bit. must do it no less
        # than after every key iteration because mutual exclusiveness
        # between keys not guaranteed. no need to do it more freq
        # because a key contains no duplicates
        already = set(itertools.chain(*Keep.values()))
        # print "\n%s getting images from %s..."%(label_default,key)
        Keep[label_default] += [fname for fname in All[key] if fname
                                not in already]
  return Keep


def merge_classes(Keep):
  more = 'Y'
  while len(Keep.keys()) > 2 and more == 'Y':
    print '%s' % (', '.join(map(str,Keep.keys())))
    if raw_input('\nMerge (more) classes? (Y/N) ') == 'Y':
      merge = [10000]
      while not all([idx < len(Keep.keys()) for idx in merge]):
        for elem in enumerate(Keep.keys()): print elem
        merge = [int(elem) for elem in raw_input("\nName class numbers from above, separated by ' ': ").split()]
      merge.sort()
      merge = [Keep.keys()[i] for i in merge]
      merge_label = '+'.join(merge)
      Keep[merge_label] = [f for key in merge
                           for f in Keep.pop(key)]
#[Keep.pop(m) for m in merge]
      count_duplicates = len(Keep[merge_label])-len(set(Keep[merge_label]))
      if count_duplicates > 0:
        print "\nWARNING! merging these classes has made %i duplicates! Removing them." % (count_duplicates)
        Keep[merge_label] = list(set(Keep[merge_label]))
    else: more = False
  return Keep, len(Keep.keys())
  

def check_mutual_exclusion(Keep, num_output):
  k = Keep.keys()
  culprits = [-1,-1]
  while culprits != []:
    Keep, num_output, culprits = check_aux(Keep, num_output, k)
  return Keep, num_output


def check_aux(Keep, num_output, k):
  for i in range(len(k)-1):
    for j in range(i+1,len(k)):
      intersection = [elem for elem in Keep[k[i]]
                      if elem in Keep[k[j]]]
      l = len(intersection) 
      if l > 0:
        print '''\nWARNING! %s and %s share %i images,
         i.e. %i percent of the smaller of the two.
         current implementation of neural net can only take one 
         label per training case, so those images will be duplicated
         and treated as differently classified training cases, which
         is suboptimal for training.'''%(k[i], k[j], l, 100*l/min(len(Keep[k[j]]),len(Keep[k[i]])))
        if raw_input('\nDo you want to continue with current setup or merge classes? (C/M) ') == 'M':
          Keep, num_output = merge_classes(Keep)
          return Keep, num_output, [Keep[k[i]], Keep[k[j]]]
  return Keep, num_output, []
      

def within_class_shuffle(Keep):
  ''' randomly shuffles the ordering of Keep[key] for each key. '''
  for key in Keep.keys():
    try:
      random.shuffle(Keep[key])
    except:
      print 'warning, within class shuffle failed'
  return Keep


def dump_to_files(Keep, data_info, task, data_dir):
  ''' This function "trusts" you. It will overwrite data lookup 
  files. '''
  dump = []
  part = [0, 0.95, 1] # partition into train val test
  dump_fnames = ['train.txt','val.txt'] #,'test.txt']
  for i in xrange(len(dump_fnames)):
    dump.append([])
    for [key,num] in [('Default',0),(task,1)]:
      l = len(Keep[key])
      dump[i] += [[f,num] for f in
                  Keep[key][int(part[i]*l):int(part[i+1]*l)]]
    # this is the important shuffle actually
    random.shuffle(dump[i])
    if os.path.isfile(oj(data_info,dump_fnames[i])):
      print "WARNING: overwriting", oj(data_info,dump_fnames[i])
    with open(oj(data_info,dump_fnames[i]),'w') as dfile:
      dfile.writelines(["%s %i\n" % (oj(data_dir,f),num)
                        for (f,num) in dump[i]])

    
def flag_lookup(labels):
  labels, flags = labels.split('-'), []
  with open('/homes/ad6813/data/flag_lookup.txt','r') as f: 
    content = f.readlines()
    for line in content:
      if line.split()[0] in labels:
        flags.append(line.split()[1])
  return flags


def print_help():
  print '''Usage eg: 
  ./setup.py ./setup.py --task=scrape --box=blue --learn=6-14'''
  if os.path.exists('/homes/ad6813'):
    print 'flags:', open('/homes/ad6813/data/flag_lookup.txt','r').readlines()
    # for line in open('/homes/ad6813/data/flag_lookup.txt','r').readlines():
    #   print line
  
  
if __name__ == '__main__':
  import sys, getopt

  if len(sys.argv) == 1:
    print_help()
  
  opts, extraparams = getopt.gnu_getopt(sys.argv[1:], "", ["task=", "box=", "learn=", "target-bad-min="])
  optDict = dict([(k[2:],v) for (k,v) in opts])
  print optDict
  
  if not "task" in optDict:
    raise Exception("Need to specify --task flag")
  task = optDict["task"]
  data_info = "/data2/ad6813/caffe/data/" + task
  
  if not "box" in optDict:
    raise Exception("Need to specify --box flag\nRed, Blue, RedBlue")
  data_dir = "/data2/ad6813/pipe-data/" + optDict["box"].capitalize() + "box/raw_data/dump"
  
  if not "learn" in optDict:
    raise Exception("Need to specify --learn flag\nlabNum1-labNum2-...-labNumk")
  pos_class = flag_lookup(optDict["learn"])

  target_bad_min = None
  if "target-bad-min" in optDict:
    target_bad_min = float(optDict["target-bad-min"])
    
  # baseDir = os.path.abspath("../task/" + task) + "/"

  # write to read file how to interpret values as classes and might
  # as well save entire command
  if not os.path.isdir(data_info): os.mkdir(data_info)
  with open(oj(data_info,'setup_history.txt'), 'w') as read_file:
    read_file.write(" ".join(sys.argv)+'\n')

  # do your shit
  main(data_dir, data_info, task, pos_class, target_bad_min)

  # still need to automate this
  # p = subprocess.Popen("./setup_rest.sh " + task + " " + str(num_output), shell=True)
  # p.wait()

  # task = 'scrape' # have already

  # GENERALISE THIS
  # avoid_flags = ['NoVisibleEvidenceOfScrapingOrPeeling','PhotoDoesNotShowEnoughOfScrapeZones','UnsuitablePhoto']
  # classification = ' 0'
  
  # using_pickle = False
  # pickle_fname = 'redbox_vacant_'+task+'_negatives.pickle'
  # redbox_dir = '/data2/ad6813/pipe-data/Redbox/raw_data/dump/'
  # fn_train = '/data2/ad6813/caffe/data/scrape/train.txt'
  
  # add_num_pos, add_num_neg = ar.same_amount_as_bluebox(data_dir, task, pos_class) # how many imgs to add
  
  # ar.bring_redbox_negatives(task, avoid_flags, classification, add_num_neg, pickle_fname, redbox_dir, fn_train, using_pickle)

  # flag = 'NoVisibleEvidenceOfScrapingOrPeeling'
  
  # # NOT RANDOM! USING TAIL
  # print 'bringing in redbox positives...'
  # ar.bring_redbox_positives(task, flag, add_num_pos, 10)

