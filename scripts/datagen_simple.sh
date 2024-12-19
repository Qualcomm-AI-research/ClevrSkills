#!/usr/bin/env bash
# set the $PROJECT_ROOT environment variable to the location of ClevrSkillsEnvSuite and
# execute this script from the root directory of the repo.

# Match pose
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/match_pose --task MatchPose \
        --task-args "{'ngoals':1}" --seed 42 --num-procs 2 --ks-wrobot --save-video

# Move without hitting
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/move_without_hitting \
        --task MoveWithoutHitting --task-args "{}" --seed 42 --num-procs 2 --ks-wrobot --save-video

# Pick
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/pick --task Pick \
        --task-args "{}" --seed 42 --num-procs 2 --save-video

# Place
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/place --task PlaceOnTop \
        --task-args "{'spawn_at_gripper':True}" --seed 42 --num-procs 2 --save-video

# Push towards
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/push --task Push \
        --task-args "{}" --seed 42 --num-procs 2 --save-video

# Touch without moving
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/touch --task Touch \
        --task-args "{}" --seed 42 --num-procs 2 --save-video

# Touch and push 
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/touch_push --task Touch \
        --task-args "{'push':True}" --seed 42 --num-procs 2 --save-video

# Touch and topple
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/touch_topple --task Touch \
        --task-args "{'topple':True}" --seed 10042 --num-procs 2 --save-video

# Throw
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/throw --task Throw \
        --task-args "{'spawn_at_gripper':True, 'target_2d':False}" --seed 42 --num-procs 2 --save-video

# Throw to topple
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/throw_topple --task Throw \
        --task-args "{'spawn_at_gripper':True, 'target_2d':False, 'topple_target':True}" --seed 10042 --num-procs 2 --save-video

# Trace
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/trace --task Trace \
        --task-args "{'ngoals':3}" --seed 42 --num-procs 2 --save-video

# Rotate
PYTHONPATH=./ python clevr_skills/clevr_skills_oracle.py --num-episodes 10 --record-dir $PROJECT_ROOT/L0/rotate --task Rotate \
        --task-args "{'num_actors':3}" --seed 42 --num-procs 2 --save-video
