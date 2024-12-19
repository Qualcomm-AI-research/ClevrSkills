# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import mplib

from clevr_skills.predicates import *
from clevr_skills.tasks import *
from clevr_skills.utils import *

from . import clevr_skills_env

if not mplib.__version__.startswith("0.2"):
    print(f"Warning: mplib version is {mplib.__version__}; version 0.2.1 is recommended")
