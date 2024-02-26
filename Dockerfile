FROM julia

RUN apt-get update && apt-get install -y xorg-dev mesa-utils xvfb libgl1 freeglut3-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev xsettingsd x11-xserver-utils
