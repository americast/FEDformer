import pudb

prob = str(input("prob? (y/n): "))

import os
import numpy as np

all_files = os.listdir("results/")
all_files = [x for x in all_files if "traffic" in x]
if prob == "n":
    all_files = [x for x in all_files if "Informer" in x]
    all_files = [x for x in all_files if "Informer_prob" not in x]
else:
    all_files = [x for x in all_files if "Informer_prob" in x]
all_files = [x for x in all_files if "pl48" in x]
all_rmse = []
all_crps = []
for file in all_files:
    f = np.load("results/"+file+"/metrics.npy")
    rmse_here = float(f[2])
    # crps_here = float(f.readline())
    all_rmse.append(rmse_here)
    # all_crps.append(crps_here)

all_rmse = [x for x in all_rmse if x > 0.3]
# all_crps = [x for x in all_crps if x > 0.3]
print(np.mean(all_rmse))
# print(np.mean(all_crps))
