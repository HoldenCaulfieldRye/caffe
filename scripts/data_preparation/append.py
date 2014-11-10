#!/usr/bin/env python

def print_help():
  print """Usage:
  ./append.py appender appendee iters"""
  sys.exit()

if __name__ == '__main__':
  import sys

  if len(sys.argv) != 4:
    print_help()

  try: n = int(sys.argv[3])
  except: print_help()

  with open(sys.argv[1],'r') as f:
    content = f.readlines()
    with open(sys.argv[2],'a') as appendee:
      for i in range(n): appendee.writelines(content)
    
