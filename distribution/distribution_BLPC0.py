import os
import sys

total_subsystems = int(sys.argv[1])

command = "singularity exec --nv -B /mnt_blpd7,/mnt_blpd1,/mnt_blpd10,/mnt_blpd11,/mnt_blpd12,/mnt_blpd13,/mnt_blpd14,/mnt_blpd15,/mnt_blpd16,/mnt_blpd17,/mnt_blpd18,/mnt_blpd19,/mnt_blpd2,/mnt_blpd3,/mnt_blpd4,/mnt_blpd5,/mnt_blpd6,/mnt_blpd7,/mnt_blpd8,/mnt_blpd9,/datax/scratch/pma  peterma-ml3 python3 /home/pma/peterma-ml/BL-Reservoir/development_env/F-Engine_Search/distribution/wrapper.py /home/pma/peterma-ml/BL-Reservoir/development_env/F-Engine_Search/GBT_pipeline full_search_dynamic_forest_BLPC0_dynamic.py"
# command = "python3 /home/pma/peterma-ml/BL-Reservoir/development_env/F-Engine_Search/distribution/wrapper.py /home/pma/peterma-ml/BL-Reservoir/development_env/F-Engine_Search/GBT_pipeline full_search_dynamic_forest_BLPC0_dynamic.py"

print("exec")
for i in range(total_subsystems):
    pre = "screen  -S instance_"+str(i)+" -d -m "
    final_command = pre+command+" "+str(total_subsystems)+" "+str(i)
    print(final_command)
    os.system(final_command)