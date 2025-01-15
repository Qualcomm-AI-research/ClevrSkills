# ClevrSkills: Compositional Language And Visual Understanding in Robotics
<div align="center">
<a href="https://arxiv.org/abs/2411.09052"><img src='https://img.shields.io/badge/arXiv-ClevrSkills-red' alt='Paper PDF'></a>

<!-- ![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge) -->
</div>

ClevrSkills is a task suite built on [ManiSkill2](https://github.com/haosulab/ManiSkill/tree/v0.5.3) for benchmarking compositional generalization in robotics. The suite includes a curriculum of tasks with three levels of compositional understanding, starting with simple tasks requiring basic motor skills. It includes 33 tasks in total with 12 tasks in L0, 15 tasks in L1 and 6 tasks in L2 levels respectively. It also includes an accompanying dataset of ~330k successful trajectories available [here](https://www.qualcomm.com/developer/software/clevrskills-dataset) generated using oracle policies found in this repository.

<!-- update the link to public repo. -->
This repo contains code for the gym environment, the task suite and oracle policies to generate the dataset. To evaluate your policies on ClevrSkills see [clevrskills-bench](https://github.com/Qualcomm-AI-research/clevrskills-bench).

Some example tasks in L0 are shown below.

![L0 Tasks](clevr_skills/assets/images/grid_small.gif)
 
## Getting started

The code was tested on Ubuntu 22.04, Python>=3.10, Torch 2.0.1 and Docker v20.10. 

First, clone the repository.

```bash
git clone https://github.com/Qualcomm-AI-research/ClevrSkills.git
```

Build the Docker image and start a container.

```bash
cd ClevrSkills
# this will create a docker image with the tag as clevrskills:latest
docker build -f docker/Dockerfile --tag clevrskills:latest .
# run the clevrskills:latest image with /clevrskills as the working directory
docker run --rm -it -w /clevrskills --gpus=all clevrskills:latest /bin/bash
```

Once the container is up and running, download the required assets using

```bash
PYTHONPATH=./ bash scripts/download_resources.sh
```

Note that the resource will end up inside the Docker container, so when restarting the container you will have to download them again.

After downloading the resources, the (unit) tests may be run as a sanity check. 
The tests include actual rollouts for several tasks; they may take 10 to 15 minutes to complete.

```bash
pytest tests
```

If any of the resources are missing (e.g., download failed), a warning should be issued:

```text
Warning: it looks like {resources} are missing; please double check and run download_resources.sh if needed
```

If you see this warning, make sure you ran the `download_resources` script (see above) and make sure it did not run into any errors.

You are now ready to use ClevrSkills and benchmark your policies on it. See [clevrskills-bench](https://github.com/Qualcomm-AI-research/clevrskills-bench) for an example where we benchmark our StreamRoboLM model.

All the tasks in ClevrSkills (and ManiSkill2) follow the gymnasium interface. 
This [code snippet](https://gymnasium.farama.org/introduction/basic_usage/#interacting-with-the-environment) shows very concretely how a policy is supposed to interact with the environment.

## Repository structure

All the main code is placed inside `clevr_skills` directory. Below is a small description of each of the scripts / main directories:

```text
ClevrSkills
│
├───clevr_skills:
│   │
│   ├───agents: Contains code for the simulated robots supported in ClevrSkills. For
│   │   now, we support UFACTORY xArm 6 and Panda robots with vacuum grippers.
│   │
│   ├───assets: Contains all the assets (objects, robot URDFs, the background template for 
│   │   video generation, etc). It also contains the object collection splits used to generate 
│   │   train/test data. The assets downloaded by `scripts/download_resources.sh` land here.
│   │
│   ├───dataset_converters: Converters used to convert the dataset generated to fit
│   │   different baseline requirements. 
│   │  
│   ├───predicates: In ClevrSkills, a predicate is a building block for specifying tasks. 
│   │   There are both physical predicates like `on_top` (something must be on top of 
│   │   something else) and logical predicates like `sequence` (the sub-predicates in
│   │   the sequence must be fulfilled sequentially). All ClevrSkills tasks consist 
│   │   of one or more predicates. Predicates also generate reward and success. 
│   │   In this directory, we define all the predicates that can be used to define tasks. 
│   │ 
│   ├───solvers: Solvers are policies that control the robot to solve predicates. 
|   |   Once we have our tasks defined in predicates, we can use specific solvers
│   │   to solve each of the predicates automatically thus generating ground truth
│   │   data for training down-stream models.
│   │
│   ├───tasks: Contains classes for each task. `clevr_skills_env.py` internally creates
│   │   an instance of a task. A task generally involves initializing objects/textures,
│   │   and defining predicates to be solved and also defining extra meta data including
│   │   prompts, keysteps etc. which is useful in training models on the dataset.
│   │
│   ├───utils: A collection of ad hoc utils used across the repo.
│   │
│   ├───clevr_skills_env.py: The base environment class.
│   │  
│   ├───clevr_skills_oracle.py: Script generating demonstration trajectories for specific
│   │   tasks by first randomly initializing instances of the tasks and then using
│   │   solvers (oracle policies) to solve the tasks and save the resulting
│   │   trajectories. 
│   │
│   └───visualize_all_tasks.py: Script to visualize all the tasks implemented in
│       ClevrSkills to an `.mp4` video.
│
├───docker: `Dockerfile` and `requirements.txt`.
│
├───logs: Directory where logs will be written by the oracle.
│
├───scripts: A collection of miscellaneous scripts for generating data, downloading resources
│   and updating robot assets.
│
├───tests: Unit tests.
├───CHANGELOG.md: Changelog.
├───LICENSE: License under which this code may be used.
├───pyproject.toml: Project settings.
└───README.md: This README file.
```

## How to run

### Generating demonstration trajectories

Once installed, use the following command to generate three example trajectories on the Pick task.

```bash
python clevr_skills/clevr_skills_oracle.py --num-episodes 3 \
--record-dir check/pick --task Pick --task-args "{}" \
--seed 43 --num-procs 1 --save-video
```

Once the script completes execution, the generated trajectories will be available in the `check/pick/traj_*` directories. 
While most of the contents are self-explanatory, please refer to `Dataset Structure` 
section [here](https://www.qualcomm.com/developer/software/clevrskills-dataset) for a short description on each item. 
The `load_traj.py` script (see below) provides an easy interface to load the data. Whenever `--save-video` is
specified on the command-line (as in the example above), an `.mp4` visualization of the trajectory will 
be written, allowing for visual inspection.

We also provide example scripts to generate trajectories for all the tasks in [L0 - Simple Tasks](scripts/datagen_simple.sh), [L1 - Intermediate Tasks](scripts/datagen_intermediate.sh) and [L2 - Complex Tasks](scripts/datagen_complex.sh). Please inspect these scripts for an overview of the available tasks.

### How to evaluate

The benchmark consists of three levels of tasks. 
The tasks and the task parameters are given in [clevr_skills/utils/task_utils.py](clevr_skills/utils/task_utils.py) and 
can be accessed as follows:

```python
from clevr_skills.utils.task_utils import L0_tasks, L1_tasks, L2_tasks
```

The keys of the dictionaries `L0_tasks`, `L1_tasks` and `L2_tasks` are the names of all tasks.

To create the environments used in our paper on a particular level of tasks on a particular asset split,

```python
from clevr_skills.utils.task_utils import get_clevrskills_env

envs = get_clevrskills_env(L0_tasks, split="train")
``` 
where the split can take `"train"` which denotes asset split used during training and `"test"` which denotes asset split including unseen objects and textures during training. Both the splits can be found at [clevr_skills/assets/asset_splits/](clevr_skills/assets/asset_splits/).

To initialize a task with `task_name` on a particular `seed` and evaluate,

```python
from clevr_skills.utils.record_env import RecordEnv

env = envs[task_name] 

# optionally wrap it in record env to record the trajectory
env = RecordEnv(env, output_dir="/path/to/save/trajs", save_trajectory=True, save_video=True)
unw_env = env.unwrapped

obs, _ = env.reset(seed=seed, options={"record_dir": "/path/to/save/trajs/seed", "reconfigure": True})

# get prompts and prompt assets
prompts = unw_env.task.get_prompts()
prompt_assets = unw_env.task.get_prompt_assets()

for i in range(MAX_LEN):
  action = model.predict(prompt, prompt_assets, obs) # predict 
  obs, reward, done, _, info = env.step(action)

# if wrapped in RecordEnv, flush video and trajectory
env.flush_trajectory()
env.flush_video()
```

See [clevrskills-bench](https://github.com/Qualcomm-AI-research/clevrskills-bench) for a full example where we benchmark our StreamRoboLM model.

### Visualizing tasks to `mp4` video.

The script can be used to generate example trajectories of all the tasks implemented in ClevrSkills and visualize them to an `.mp4` file. 
The following example generates trajectories for all the tasks in L0 and visualizes them in a grid, as seen in animation at the top of this README.

```bash
python clevr_skills/visualize_all_tasks.py --output-dir vis_L0/ \
--compose-grid --grid-width 3 --grid-num-frames 200 --levels 0
```

It can also be used to visualize tasks individually. For example, the following command generates and visualize a trajectory for the Rotate task.

```bash
python clevr_skills/visualize_all_tasks.py --output-dir vis_rotate/ \
--pretty-video --tasks Rotate --prompt-visualization multi_modal
```

### Loading a trajectory

The file [`clevr_skills/utils/load_traj.py`](clevr_skills/utils/load_traj.py) implements a utility class that makes it easy to load data from a ClevrSkills trajectory.
Following snippet shows an example of how the `Traj` class can be used to load a trajectory and fetch actions and multi-modal prompts. 
This can serve as a starting point for your dataloader to train models on ClevrSkills data.

```python
from clevr_skills.utils.load_traj import Traj

example_traj = Traj("/path/to/trajectory")
actions = example_traj.get_actions()
prompts = example_traj.get_multi_modal_prompts()
```


There is also a function to recursively scan a directory tree for trajectories:

```python
from clevr_skills.utils.load_traj import Traj, find_trajectories

for traj_path in find_trajectories("/path/to/dataset")
    traj = Traj(traj_path)
```

For more details, please refer to [`clevr_skills/utils/load_traj.py`](clevr_skills/utils/load_traj.py).

### Sapien Viewer visualization

Both `clevr_skills_oracle.py` and `visualize_all_tasks.py` will open an interactive window when `--enable-sapien-viewer` is passed
on the command-line. The [Sapien Viewer](https://sapien.ucsd.edu/docs/2.2/tutorial/basic/viewer.html) 
shows an interactive visualization of the scene and allows the user to inspect many aspects of it.
Also, the execution of the policy can be paused. Pausing the scene is recommended during interaction with the viewer 
because otherwise graphics and motion can be choppy.

For Sapien Viewer to work, an X11 display with Vulkan support must be available. You must set the `DISPLAY` environment 
to point to a valid display. See the troubleshooting section below for a workaround when an X11 display is not available.


## Troubleshooting

### Docker
 
When installing Docker in a Linux environment, be sure to install Docker *Engine*, not Docker *Desktop*.

Docker Desktop does not support GPUs inside the container; when starting your container with `--gpus=all`, you
will get an error along the lines of `OCI runtime create failed`.

### Vulkan driver

One of the following errors may occur indicating that the vulkan driver might be broken:

```
RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed

Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.

Segmentation fault (core dumped)
```

Please follow [the instructions from ManiSkill authors](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) to fix this.

### Missing resources

Please make sure to download the resources as suggested in [Getting Started](#getting-started) if you run into a filesystem error similar to the following error

```
RuntimeError: filesystem error: cannot make canonical path: No such file or directory [.../ClevrSkills/clevr_skills/assets/vima_textures/wood_light.png]
```

### X11 display

When no X11 display is available, the Sapien viewer can't be used.
Except for this the environment should work fine, although you will 
see this error / warning from the  Sapien engine. It can be safely ignored:

```
[svulkan2] [error] GLFW error: X11: The DISPLAY environment variable is missing
[svulkan2] [warning] Continue without GLFW.
```

If you want an X11 display anyway, you'll need to set your DISPLAY environment variable to a valid display.

See general guides [here](https://www.baeldung.com/linux/docker-container-gui-applications) and
[here](https://www.baeldung.com/linux/no-x11-display-error#bd-starting-a-remote-x-client) on how to connect
ClevrSkills (running inside a Docker container ) to your X11 display outside of the Docker.

We offer two specific recipies below:

#### X11 display through Xauth

This recipe provides the proper way to connect an application (running inside of a Docker container) to
an X11 display outside the Docker container:

1. Verify that X apps work from your console:

    ```bash
    echo $DISPLAY  # should be ":0" or similar to: "localhost:10.0"
    sudo bash -c "apt update; apt install -qqy xauth xterm"
    xterm
    ```
    This should show a graphical terminal window. Close this window by entering `exit`.

2. Create a custom X authority file, for allowing the container access to the X server:

    ```bash
    XAUTHFILE=/tmp/.docker.xauth.$USER
    touch $XAUTHFILE
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTHFILE nmerge -
    ```

3. Add these command line options to your `docker` command that starts a ClevrSkills container:

    ```text
    --network host
    --env DISPLAY=$DISPLAY
    --volume $XAUTHFILE:/root/.Xauthority
    ```

4. If you connect through SSH to the host where you run your ClevrSkills container then make sure that the host's SSH daemon allows X11 port forwarding. Enable or add this line (using sudo):

    ```text
    /etc/ssh/sshd_config: X11Forwarding yes  # Specifies whether X11 forwarding is permitted. The default is yes.
    ```

    Then restart the SSH daemon:

    ```bash
    sudo systemctl restart ssh
    ```
   
    And pass the `-X` flag (or if trusted: `-Y`) to your SSH connect command.

#### X11 display through Xvfb and VNC

In some configurations, a physical display may not be available; or you might not have `sudo` rights to
follow the Xauth recipe above. In those cases, the following recipe has worked for us to get a virtual display.

Run the following commands, inside the ClevrSkills docker container:

```
export DISPLAY=21 # or any other available port
# Start X virtual frame buffer inside the Docker container:
sudo Xvfb $DISPLAY -screen 0 1920x1080x24 &
# Start VNC server inside the Docker container:
sudo x11vnc -passwd PASSWORD -display $DISPLAY -N -forever &
```

And then on your client machine, connect to the VNC server using your favorite VNC viewer.
In the example the port for the VNC viewer would be 5921.

### Sapien render engine / visualization

- When running multiple environments in parallel, you will see this harmless warning:

```
[svulkan2] [warning] A second renderer will share the same internal context with the first one. Arguments passed to constructor will be ignored.
```

- Interaction with the Sapien Viewer can be choppy when an oracle policy is running. For example when moving the camera
viewpoing around the camera viewpoint can jump around. To avoid this, first pause the viewer (see checkmark button, upper-left corner of window).


## Code formatting

The code was formatted using:

```bash
python3 -m black --safe --line-length 100 .
python3 -m isort --profile black .
```

## Citation

If you find our code useful, please cite:

```
@article{haresh2024clevrskills,
  title={ClevrSkills: Compositional Language And Visual Understanding in Robotics},
  author={Haresh, Sanjay and Dijkman, Daniel and Bhattacharyya, Apratim and Memisevic, Roland},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```
