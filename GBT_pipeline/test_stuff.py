import sys
total_subsystems = int(sys.argv[1])
id_subsystem = int(sys.argv[2])

count = 0
while True:
    count+=1
    if count%10000:
        print(count)