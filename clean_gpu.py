import os
import cmd

cmd = 'sudo fuser -v /dev/nvidia* > fsuer_info.txt'

os.system(cmd)

with open('fsuer_info.txt', 'r') as f:
    line = f.readlines()[0].strip().split(' ')
    for pid in line:
        cmd = 'sudo kill -9 ' + pid
        os.system(cmd)
