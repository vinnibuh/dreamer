FROM tensorflow/tensorflow:latest-gpu
WORKDIR /app

RUN apt-get update
RUN apt-get install -y curl \
	git \
	unzip \
	wget \
	vim \
	ffmpeg \
	libglew-dev
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
RUN pip3 install tensorflow_probability
RUN pip3 install pandas
RUN pip3 install wandb
RUN pip3 install matplotlib
RUN pip3 install gym

RUN pip3 install git+git://github.com/deepmind/dm_control.git
COPY . .
