from time import time
import datetime


def intro(n):
    header = '''
======================================================
   _____ ________________   __  _____ 
  / ___// ____/_  __/  _/  /  |/  / / 
  \__ \/ __/   / /  / /   / /|_/ / /  
 ___/ / /___  / / _/ /   / /  / / /___
/____/_____/ /_/ /___/  /_/  /_/_____/

    '''
    print(header)
    print("Author: Peter Xiangyuan Ma")
    print("Date: "+str(datetime.datetime.now()))
    print("Execution on: "+str(n)+" files")
    print('======================================================')
