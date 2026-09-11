#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# set the $PROJECT_ROOT environment variable to the location of ClevrSkillsEnvSuite and
# execute this script from the root directory of the repo.

mkdir -p "$PROJECT_ROOT/memory_tasks"

# rotate restore
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir "$PROJECT_ROOT/memory_tasks/rotate_restore" --task Rotate \
        --task-args "{'num_actors':3,'restore':True,'split':'cs_mem'}" --num-steps-done 10 --seed 42 --num-procs 2 --save-video

# stack topple
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir "$PROJECT_ROOT/memory_tasks/stack_topple" --task SingleStack \
        --task-args "{'num_actors':4,'topple':True,'split':'cs_mem'}" --num-steps-done 10 --seed 2178 --num-procs 2 --save-video
# swap
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir "$PROJECT_ROOT/memory_tasks/swap" --task Swap \
        --task-args "{'split':'cs_mem'}" --seed 42 --num-procs 2 --save-video
# touch pick
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir "$PROJECT_ROOT/memory_tasks/touch_pick" --task TouchPick \
        --task-args "{'split':'cs_mem'}" --num-steps-done 5 --seed 42 --num-procs 2 --save-video

# place-next-to-and-restore
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir "$PROJECT_ROOT/memory_tasks/place_next_restore" --task PlaceNextTo \
        --task-args "{'split':'cs_mem', 'restore':True}" --num-steps-done 5 --seed 42 --num-procs 2 --save-video
