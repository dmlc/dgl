import psutil, os
import sys

def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    print (parent)
    for child in parent.children(recursive=True):
        print (child)
        child.kill()
    if including_parent:
        parent.kill()

me = int(sys.argv[1])
for i in range(8):
    try:
        kill_proc_tree(me+i)
    except:
        continue
