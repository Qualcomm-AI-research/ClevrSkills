#!/usr/bin/env bash
# set the $PROJECT_ROOT environment variable to the location of ClevrSkillsEnvSuite and
# execute this script from the root directory of the repo.

# follow order
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/follow_order --task FollowOrder \
        --task-args "{'num_areas':4,'num_predicates':4}" --seed 42 --num-procs 2 --save-video

# follow order restore
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 3 --record-dir $PROJECT_ROOT/L1/follow_order_restore --task FollowOrder \
        --task-args "{'num_areas':2,'num_predicates':2,'restore':True}" --seed 10042 --num-procs 2 --save-video

# neighbour
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/neighbour --task Neighbour \
        --task-args "{'num_areas':3}" --seed 121 --num-procs 2 --save-video

# novel noun
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/novel_noun --task NovelNoun \
    --task-args "{'num_actors':3,'num_areas':3}" --seed 43 --num-procs 2 --save-video

# novel adjective
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/novel_adj --task NovelAdjective \
    --task-args "{}" --seed 43 --num-procs 2 --save-video

# novel noun and adjective
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/novel_adj_noun --task NovelNounAdjective \
    --task-args "{}" --seed 43 --num-procs 2 --save-video

# rearrange
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/rearrange --task Rearrange \
        --task-args "{'num_actors':3}" --seed 42 --num-procs 2 --save-video

# rearrange restore
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/rearrange_restore --task Rearrange \
        --task-args "{'num_actors':3,'restore':True}" --seed 11000 --num-procs 2 --save-video

# rotate restore
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/rotate_restore --task Rotate \
        --task-args "{'num_actors':3,'restore':True}" --seed 901201 --num-procs 2 --save-video

# rotate symmetry
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/rotate_symmetry --task RotateSymmetry \
        --task-args "{'num_actors':4}" --seed 6550634 --num-procs 2 --save-video

# simple_manipulation
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/simple_manipulation --task PlaceOnTop \
        --task-args "{'num_actors':3,'num_areas':3}" --seed 100042 --num-procs 2 --save-video

# stack
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/single_stack --task SingleStack \
        --task-args "{'num_actors':4}" --seed 42 --num-procs 2 --save-video

# sort
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/sort2d --task Sort2d \
        --task-args "{'num_actors':4,'num_areas':3}" --seed 42 --num-procs 2 --save-video

# stack reversed
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/stack_reversed --task SingleStack \
        --task-args "{'num_actors':4,'reverse':True}" --seed 9958614 --num-procs 2 --save-video

# # Swap
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L1/swap --task Swap \
        --task-args "{}" --seed 42 --num-procs 2 --save-video
