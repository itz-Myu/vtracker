#!/bin/bash
source tracker_env/bin/activate
export DISPLAY=:0
export XDG_SESSION_TYPE=x11
export EGL_PLATFORM=x11
export __GLX_VENDOR_LIBRARY_NAME=nvidia
python main.py "$@"
