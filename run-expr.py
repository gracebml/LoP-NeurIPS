# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # LoP-NeurIPS Experiments
# 
# This notebook is designed to run the LoP second-order optimizer experiments in Kaggle's offline environment.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-18T20:58:13.071284Z","iopub.execute_input":"2026-04-18T20:58:13.071351Z","iopub.status.idle":"2026-04-18T20:58:15.536050Z","shell.execute_reply.started":"2026-04-18T20:58:13.071341Z","shell.execute_reply":"2026-04-18T20:58:15.535757Z"}}
# 1. Install required packages from the offline dataset `lop-packages`
!pip install --no-index --find-links /kaggle/input/datasets/mlbang/lop-packages mlproj-manager tqdm

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-18T20:58:15.536741Z","iopub.execute_input":"2026-04-18T20:58:15.536869Z","iopub.status.idle":"2026-04-18T20:58:16.840181Z","shell.execute_reply.started":"2026-04-18T20:58:15.536857Z","shell.execute_reply":"2026-04-18T20:58:16.839954Z"}}
# 2. Extract and layout the source code in /kaggle/working
import os
import shutil

INPUT_SRC = '/kaggle/input/datasets/mlbang/lop-src' 
WORKING_DIR = '/kaggle/working/lop-src'

if not os.path.exists(WORKING_DIR):
    shutil.copytree(INPUT_SRC, WORKING_DIR)
    print(f"Copied source code to {WORKING_DIR}")

os.chdir(WORKING_DIR)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-18T20:58:16.840570Z","iopub.execute_input":"2026-04-18T20:58:16.840657Z","iopub.status.idle":"2026-04-18T20:58:16.951804Z","shell.execute_reply.started":"2026-04-18T20:58:16.840648Z","shell.execute_reply":"2026-04-18T20:58:16.951581Z"}}
# 3. Setup Dataset links
os.makedirs("data/cifar-100-python", exist_ok=True)

# 3a. CIFAR-100 (Attached as cifar-100-python)
if os.path.exists("/kaggle/input/datasets/mlbang/cifar100"):
    !ln -sf /kaggle/input/datasets/mlbang/cifar100/* ./data/cifar-100-python/
    print("Linked CIFAR-100 dataset to data/cifar-100-python/")

# %% [code] {"execution":{"iopub.status.busy":"2026-04-18T20:58:16.952169Z","iopub.execute_input":"2026-04-18T20:58:16.952251Z","iopub.status.idle":"2026-04-18T20:58:57.382078Z","shell.execute_reply.started":"2026-04-18T20:58:16.952241Z","shell.execute_reply":"2026-04-18T20:58:57.381805Z"}}
# 1. CIFAR-100 Experiment
!python main.py cifar -c lop/incremental_cifar/cfg/sassha_sdp.json --index 0

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-18T20:58:57.382586Z","iopub.execute_input":"2026-04-18T20:58:57.382703Z","iopub.status.idle":"2026-04-18T20:58:57.384335Z","shell.execute_reply.started":"2026-04-18T20:58:57.382691Z","shell.execute_reply":"2026-04-18T20:58:57.384174Z"}}
# 2. RL Experiment (Ant-v4)

# !python lop/rl/run_ppo_2nd.py -c lop/rl/cfg/ant/sassha_sdp.yml -s 1

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-18T20:58:57.384909Z","iopub.execute_input":"2026-04-18T20:58:57.385042Z","iopub.status.idle":"2026-04-18T20:58:57.396741Z","shell.execute_reply.started":"2026-04-18T20:58:57.385034Z","shell.execute_reply":"2026-04-18T20:58:57.396590Z"}}
# 3. Permuted MNIST Experiment

# !python main.py mnist -c lop/permuted_mnist/cfg/bp.json

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-18T20:58:57.396975Z","iopub.execute_input":"2026-04-18T20:58:57.397042Z","iopub.status.idle":"2026-04-18T20:58:57.407041Z","shell.execute_reply.started":"2026-04-18T20:58:57.397035Z","shell.execute_reply":"2026-04-18T20:58:57.406899Z"}}
# 4. ImageNet Experiment

# !python main.py imagenet -c lop/imagenet/cfg/bp.json

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-18T20:58:57.407282Z","iopub.execute_input":"2026-04-18T20:58:57.407371Z","iopub.status.idle":"2026-04-18T20:58:57.630219Z","shell.execute_reply.started":"2026-04-18T20:58:57.407364Z","shell.execute_reply":"2026-04-18T20:58:57.629923Z"}}
# 5. Check outputs
!ls -lah results/
!ls -lah results/rl_2nd/

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-04-18T20:58:57.630647Z","iopub.execute_input":"2026-04-18T20:58:57.630771Z","iopub.status.idle":"2026-04-18T20:58:57.632233Z","shell.execute_reply.started":"2026-04-18T20:58:57.630760Z","shell.execute_reply":"2026-04-18T20:58:57.632060Z"}}
# 6. Zip Results for Download
# !zip -r results_run.zip results/