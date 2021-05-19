from preprocess import get_data
from build_model import multi_gpu_load
import sys
import warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

cadence_set = ['../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_58929_GJ380_fine.h5',
                "../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_59291_HIP48887_fine.h5",
                "../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_59650_GJ380_fine.h5",
                "../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_60004_HIP48924_fine.h5",
                "../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_60354_GJ380_fine.h5",
                "../../../../../../../mnt_blpd7/datax2/dl/GBT_57636_60706_HIP48954_fine.h5"
                ]
                
preprocess = get_data(cadence_set, 1200,1500)

model = multi_gpu_load()