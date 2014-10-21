import os, sys
from os.path import join as ojoin

# usage: python create_deploy_prototxt.py model-dir=.. data-dir=..

# Note! data-dir should be data/<name>, not data/<name>/test


def get_train_file(model_dir):
  train_file = ''
  for fname in os.listdir(model_dir):
    if fname.endswith('train.prototxt'):
      return open(ojoin(model_dir,fname),'r')
  if not os.path.isdir(model_dir):
    print "error:", model_dir, "does not exist"
  else: print 'no train prototxt found in', model_dir
  sys.exit()

def edit_train_content_for_deploy(content, oversample=True):
  if oversample: mult = 10
  else: mult = 1
  idx = 0
  while idx < len(content):
    if 'name: ' in content[idx] and 'FineNet' in content[idx]:
      content[idx] = 'name: "CaffeNet"\n'
      idx += 1
    if 'name: "data"' in content[idx]:
      content[idx-1] = 'input: "data"\n'
      content[idx] = 'input_dim: %i\n'%(10) 
      content[idx+1] = 'input_dim: 3\n'
      content[idx+2] = 'input_dim: 227\n'
      content[idx+3] = 'input_dim: 227\n'
      del content[idx+4:idx+12]
      idx += 4
    elif 'blobs_lr' in content[idx]:
      del content[idx]
      # idx += 1
    elif 'weight_decay' in content[idx]:
      del content[idx]
      # idx += 1
    elif 'weight_filler' in content[idx]:
      del content[idx:idx+8]
    elif 'accuracy' in content[idx]:
      del content[idx-1:idx+5]
    elif 'name: "loss"' in content[idx]:
      content[idx] =   '  name: "prob"\n'
      content[idx+1] = '  type: SOFTMAX\n'
      content[idx+3] = '  top: "prob"\n'
      break
    else:
      idx += 1
  return content

def write_content_to_deploy_file(model_dir, content):
  model_name = model_dir.split('/')[-1]
  model_name = model_name.split('-fine')[0]
  fname = ojoin(model_dir,model_name+'_deploy.prototxt')
  # print "fname: %s"%(fname)
  deploy_file = open(fname,'w')
  deploy_file.writelines(content)
  deploy_file.close()

if __name__ == '__main__':

  for arg in sys.argv:
    if "model-dir=" in arg:
      model_dir = os.path.abspath(arg.split('=')[-1])
    elif "data-dir=" in arg:
      data_dir = os.path.abspath(arg.split('=')[-1])
  
  train_file = get_train_file(model_dir)
  # num_imgs = len(os.listdir(ojoin(data_dir,'test')))
  content = train_file.readlines()
  content = edit_train_content_for_deploy(content)
  write_content_to_deploy_file(model_dir, content)
    



