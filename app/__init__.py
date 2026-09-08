"""BAS Crew Activity Recognition.

MediaPipe and TensorFlow Lite print a wall of native warnings on import -
XNNPACK delegates, feedback-manager notices - that mean nothing to someone
running this and look like something has gone wrong. They are quietened here,
before any submodule imports them, so the console output stays readable.
"""

import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
