#!/usr/bin/env bash
# set the $PROJECT_ROOT environment variable to the location of ClevrSkillsEnvSuite and
# execute this script from the root directory of the repo.

# Balance
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L2/balance --task BalanceScale \
        --task-args "{'num_actors':4}" --seed 42 --num-procs 2 --save-video

# SortStack
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L2/sort_stack --task SortStack \
        --task-args "{'num_actors':6,'num_areas':3}" --seed 2234489 --num-procs 2 --save-video

# Stack and topple
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L2/stack_topple --task SingleStack \
        --task-args "{'num_actors':4,'topple':True}" --seed 42 --num-procs 2 --save-video

# Throw and sort
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L2/throw_sort --task ThrowAndSort \
        --task-args "{'num_actors':4,'num_areas':3}" --seed 28108777 --num-procs 2 --save-video

# Swap push
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L2/swap --task Swap \
        --task-args "{'push':True}" --seed 11977685 --num-procs 2 --save-video

# Swap rotate
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L2/swap_rotate --task SwapRotate \
        --task-args "{}" --seed 4304572 --num-procs 2 --save-video
