#!/usr/bin/env python
import numpy as np
import os
from os.path import join as oj
from PIL import Image
from operator import itemgetter as ig
import itertools 
import datetime
from shutil import rmtree
import random, subprocess
import add_redboxes as ar

# expected input: ./setup.py --task= --box= --learn= [--target-bad-min=]

# ./setup.py --task=unsuit --box=blue --learn=3-10-14

def main(data_dir, data_info, task, pos_class, u_bad_min=None):
  ''' This is the master function. data_dir: where raw data is. data_info: where to store .txt files. '''
  Keep = get_label_dict_knowing(data_dir, task, pos_class)
  if u_bad_min is not None:
    Keep = under_sample(Keep, u_bad_min)
  Keep = within_class_shuffle(Keep)
  print 'finished shuffling'
  dump_to_files(Keep, data_info, task, data_dir)
  return len(Keep[task]), len(Keep['Default'])


def get_label_dict_knowing(data_dir, task, pos_class):
  ''' get_label_dict() knowing exactly which flags to look for and 
  how to group them into classes. 
  task is the name of what we're learning to detect,
  pos_class is a list of the actual flag names to look for. '''
  d = {'Default': [], task: []}
  print 'generating specific dict of class:files from %s with pos class %s...'%(data_dir,pos_class)
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




def under_sample(Keep, u_bad_min):
  '''if u_bad_min not given, prompts user for one; 
  and implements it. Note that with >2 classes, this can be 
  implemented either by downsizing all non-minority classes by the
  same factor in order to maintain their relative proportions, or 
  by downsizing as few majority classes as possible until
  u_bad_min achieved. We can assume that we care mostly about 
  having as few small classes as possible, so the latter is 
  implemented.'''
  u_bad_min = float(u_bad_min)
  # minc is class with minimum number of training cases
  ascending_classes = sorted([(key,len(Keep[key]))
                              for key in Keep.keys()],
                             key=lambda x:x[1])
  maxc, len_maxc = ascending_classes[-1][0], ascending_classes[-1][1]
  minc, len_minc = ascending_classes[0][0], ascending_classes[0][1]
  total_num_images = sum([len(Keep[key]) for key in Keep.keys()])
  # print ascending_classes
  # print "\ntotal num images: %i"%(total_num_images)
  maxc_proportion = float(len_maxc)/total_num_images
  print 'maxc_proportion: %.2f, u_bad_min: %.2f'%(maxc_proportion, u_bad_min)
  if maxc_proportion > u_bad_min:
    delete_size = int((len_maxc - (u_bad_min*total_num_images))/(1-u_bad_min))
    random.shuffle(Keep[maxc])
    print '%s has %i images so %i will be randomly removed'%(maxc, len_maxc, delete_size)
    del Keep[maxc][:delete_size]
  elif maxc_proportion < u_bad_min:
    print 'woah, you want to INCREASE class imbalance!'
    delete_size = int(total_num_images - (len_maxc/float(u_bad_min)))
    random.shuffle(Keep[minc])
    print '%s has %i images so %i will be randomly removed'%(minc, len_minc, delete_size)
    del Keep[minc][:delete_size]
  assert u_bad_min == round(float(len(Keep[maxc])) / (len(Keep[maxc])+len(Keep[minc])), 2)
  for key in Keep.keys(): print key, len(Keep[key])
  return Keep


def o_sample_how_many(num_pos, num_neg, o_bad_min):
  o_bad_min = float(o_bad_min)
  print 'o_bad_min, num_pos, num_neg', o_bad_min, num_pos, num_neg
  full_copies = (1-o_bad_min)/o_bad_min
  full_copies *= float(num_neg)
  last_copy = int(full_copies) % int(num_pos)
  full_copies /= float(num_pos)
  # last_copy = (full_copies - int(full_copies)) * num_neg
  full_copies, last_copy = int(full_copies)-1, int(last_copy)
  print "oversample positives %i times plus %i extras"%(full_copies,last_copy)
  return int(full_copies), last_copy


def over_sample(fn_train, full_copies, last_copy):
  cmd = "./over_sample.sh " + fn_train + " " + str(full_copies) + " " + str(last_copy)
  p = subprocess.Popen(cmd, shell=True)
  p.wait()


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
  part = [0, 0.85, 1] # partition into train val test
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


def add_redboxes(target_bad_min, b_imbal, pos_class, task,
                 avoid_flags, redbox_dir, fn_train):
  blue_c_imb = float(target_bad_min)
  # GOING FOR NO INFO GAIN!
  add_num_pos, add_num_neg = ar.blur_no_infogain(blue_c_imb, redbox_dir, task, pos_class)
  ar.bring_redbox_negatives(task, avoid_flags, add_num_neg, redbox_dir, fn_train)

  # NOT RANDOM! USING TAIL
  print 'bringing in %i redbox positives...'%(add_num_pos)
  ar.bring_redbox_positives(task, pos_class, add_num_pos, redbox_dir, fn_train)
  num_pos, num_neg = ar.shuffle_file(fn_train)
  return num_pos, num_neg


def print_help():
  print '''Usage eg: 
  ./setup.py --task=scrape --box=blue --learn=6-14 --u-sample=0.90 --o-sample=0.55 --b-imbal=0.5'''
  if os.path.exists('/homes/ad6813'):
    # print 'flags:', open('/homes/ad6813/data/flag_lookup.txt','r').readlines()
    lines = open('/homes/ad6813/data/flag_lookup.txt','r').readlines()
    lines = [line.strip() for line in lines]
    for line in lines:
      print line
  
  
if __name__ == '__main__':
  import sys, getopt

  if len(sys.argv) == 1:
    print_help()
  
  opts, extraparams = getopt.gnu_getopt(sys.argv[1:], "", ["task=", "box=", "learn=", "u-sample=", "o-sample=", "b-imbal="])
  optDict = dict([(k[2:],v) for (k,v) in opts])
  print optDict
  
  if not "task" in optDict:
    raise Exception("Need to specify --task flag")
  task = optDict["task"]
  data_info = os.path.abspath("../../data") +'/'+task
  
  if not "box" in optDict:
    raise Exception("Need to specify --box flag\nred, blue, redblue")
  data_dir = "/data/ad6813/pipe-data/" + optDict["box"].capitalize() + "box"
  
  if not "learn" in optDict:
    raise Exception("Need to specify --learn flag\nlabNum1-labNum2-...-labNumk")
  pos_class = flag_lookup(optDict["learn"])

  u_bad_min = None
  if "u-sample" in optDict:
    u_bad_min = float(optDict["u-sample"])
    
  # save entire command
  if not os.path.isdir('../../data/'+task): os.mkdir('../../data/'+task)
  date = '_'.join(str(datetime.datetime.now()).split('.')[0].split())
  with open(data_info+'/setup_history_'+date+'.txt', 'w') as read_file:
    read_file.write(" ".join(sys.argv)+'\n')

  # do your shit
  num_pos, num_neg = main(data_dir, data_info, task, pos_class, u_bad_min)

  fn_train = data_info+'/train.txt'

  # GENERALISE THIS
  if 'b-imbal' in optDict:
    b_imbal = float(optDict["b-imbal"])
    avoid_flags = [] # dont need cos Raz moved away unsuitables?
    redbox_dir = '/data/ad6813/pipe-data/Redbox'

    # How many redboxes to add:
    # add_num_pos, add_num_neg = ar.same_amount_as_bluebox(data_dir, task, pos_class)
    if u_bad_min == None:
      message = '''ERROR: no u_bad_min given to maintain  
      class imbalance after redbox additions. 
      If you don't want any undersampling and still want to add 
      redbox st imbalance unchanged, then yeah, you need to 
      implement that.'''
      raise Exception(message)

    num_pos, num_neg = add_redboxes(u_bad_min, b_imbal, pos_class, task,avoid_flags, redbox_dir, fn_train)
    

  # oversampling
  if "o-sample" in optDict:
    o_bad_min = float(optDict["o-sample"])
    full_copies, last_copy = o_sample_how_many(num_pos, num_neg, o_bad_min)
    over_sample(fn_train, full_copies, last_copy)
    
  # setup task/etc
  p = subprocess.Popen("./rest_setup.sh " + task, shell=True)
  p.wait()






