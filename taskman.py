
# coding: utf-8

# In[ ]:

from datetime import datetime
from os import listdir
from os.path import isfile, join
import os
import subprocess
import time

def run(command):
    output = subprocess.check_output(command, shell=True)
    return output

if __name__ == '__main__':
    path = '.'
    cfs = sorted([join(path, f) for f in listdir(path) if f.startswith('p_') and isfile(join(path, f))])
    with open('taskman.txt', 'a') as logf:
        for cf in cfs:
#             cmd = 'oarsub -l core=1,walltime=120 -t moonshot --notify "mail:daoyuan.li@uni.lu" --array-param-file %s segmentation.sh' % cf
            cmd = 'oarsub -l core=1,walltime=10 -t moonshot --array-param-file %s segmentation.sh' % cf
            rerun = True
            while rerun:
                logf.write('%s\n%s\n' % (str(datetime.now()), cmd))
                logf.flush()
                output = ''
                print '%s\n%s' % (str(datetime.now()), cmd)
                try:
                    output = run(cmd)
                except:
                    pass
                logf.write('%s' % output)
                if 'OAR_ARRAY_ID=' in output:
                    rerun = False
                    try:
                        os.remove(cf)
                        logf.write('Tasks in queue. Removed param file: %s\n\n' % cf)
                    except:
                        pass
                else:
                    time.sleep(60)

