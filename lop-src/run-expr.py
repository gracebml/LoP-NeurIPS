# %% [markdown]
# # LoP-NeurIPS Experiments

# This notebook is designed to run the LoP second-order optimizer experiments in Kaggle's offline environment.

# %% [code]
# 1. Install required packages from the offline dataset `lop-packages`
!pip install --no-index --find-links /kaggle/input/lop-packages/ mlproj-manager tqdm

# %% [code]
# 2. Extract and layout the source code in /kaggle/working
import os
import shutil

INPUT_SRC = '/kaggle/input/lop-src' 
WORKING_DIR = '/kaggle/working/lop-src'

if not os.path.exists(WORKING_DIR):
    shutil.copytree(INPUT_SRC, WORKING_DIR)
    print(f"Copied source code to {WORKING_DIR}")

os.chdir(WORKING_DIR)

# %% [code]
# 3. Setup Dataset links
os.makedirs("data", exist_ok=True)

# 3a. CIFAR-100 (Attached as cifar-100-python)
if os.path.exists("/kaggle/input/cifar-100-python"):
    !ln -sf /kaggle/input/cifar-100-python/* ./data/
    print("Linked CIFAR-100 dataset.")

# %% [markdown]
# 1. CIFAR-100 Experiment
!python main.py cifar -c lop/incremental_cifar/cfg/sassha_sdp.json --index 0

# %% [code]
# 2. RL Experiment (Ant-v4)

# !python lop/rl/run_ppo_2nd.py -c lop/rl/cfg/ant/sassha_sdp.yml -s 1

# %% [code]
# 3. Permuted MNIST Experiment

# !python main.py mnist -c lop/permuted_mnist/cfg/bp.json

# %% [code]
# 4. ImageNet Experiment

# !python main.py imagenet -c lop/imagenet/cfg/bp.json

# %% [code]
# 5. Check outputs
!ls -lah results/
!ls -lah results/rl_2nd/

# %% [code]
# 6. Zip Results for Download
# !zip -r results_run.zip results/
