import sys
import os

cd = "cd "+str(sys.argv[1])
executable = str(sys.argv[2])
total = str(sys.argv[3])
id_tag =  str(sys.argv[4])
os.system(cd)
os.system("python3 "+executable+" "+total+" "+id_tag)