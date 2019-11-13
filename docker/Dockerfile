FROM shadowrobot/dexterous-hand:kinetic-night-build

LABEL Description="This ROS Kinetic image contains Shadow's dexterous hand software with build tools. It includes IDE environments. Nvidia compatible" Vendor="Shadow Robot" Version="1.0"

RUN apt-get update \
&& apt-get install -y --no-install-recommends pkg-config libxau-dev libxdmcp-dev libxcb1-dev libxext-dev libx11-dev \
&& rm -rf /var/lib/apt/lists/*

# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 /usr/local/lib/x86_64-linux-gnu /usr/local/lib/x86_64-linux-gnu

# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04 /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json

RUN echo '/usr/local/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf \
&& ldconfig \
&& echo '/usr/local/$LIB/libGL.so.1' >> /etc/ld.so.preload \
&& echo '/usr/local/$LIB/libEGL.so.1' >> /etc/ld.so.preload


# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# install GLX-Gears
RUN apt update && apt install -y --no-install-recommends mesa-utils x11-apps && rm -rf /var/lib/apt/lists/*

# nvidia-docker hooks
LABEL com.nvidia.volumes.needed="nvidia_driver"

ENV PATH /usr/local/nvidia/bin:${PATH}

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
