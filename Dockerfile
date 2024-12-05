# FROM continuumio/anaconda3 as base
FROM nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04 as base

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# gcc und g++? komisch
# libosmesa6-dev nötig?
RUN apt update && apt install -y \
                software-properties-common \
                vim \
                wget \
                python3-dev \
                #linux-headers-$(uname -r) \
                #gcc \
                #g++ \
                #libxml2-dev \
                #libosmesa6-dev \
                #freeglut3-dev \
                gcc \
                git
       # && rm -rf /var/lib/apt/lists/*

#RUN wget https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/linux-hwe-6.2/6.2.0-37.38~22.04.1/linux-hwe-6.2_6.2.0.orig.tar.gz && mkdir /opt/linux-headers && tar -xf linux-hwe-6.2_6.2.0.orig.tar.gz -C /opt/linux-headers/
#RUN wget https://files.pythonhosted.org/packages/4d/ec/bb298d36ed67abd94293253e3e52bdf16732153b887bf08b8d6f269eacef/evdev-1.4.0.tar.gz && mkdir /opt/evdev-source && tar -xf evdev-1.4.0.tar.gz -C /opt/evdev-source/
#RUN ls /opt/linux-headers
#RUN cd /opt/evdev-source/evdev-1.4.0 && python setup.py build build_ecodes --evdev-headers /opt/linux_headers/linux-6.2/include/uapi/linux/input.h:/opt/linux_headers/linux-6.2/include/uapi/linux/input-event-codes.h

RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

## ENV STAGE
# abgekapselt, um schneller mit buildx Cache
# zu bauen, für Endprodukt aber irrelevant
FROM base as env

WORKDIR /root
RUN git clone --recursive https://github.com/aikkala/user-in-the-box.git
        #&& rm -rf banmo/.git

WORKDIR /root/user-in-the-box
# pycocotools 2.0.4: problem mit cython>3

RUN conda init 

RUN conda create -n uitb-sim2vr python=3.11 \
        && conda clean --yes --all \
        && echo "conda activate uitb-sim2vr" >> ~/.bashrc
RUN apt update && apt install -y libgl1 libgl1-mesa-glx libosmesa6 libglfw3-dev libgles2-mesa-dev

# neue RUN Befehle sollen conda env verwenden!
# conda activate hat im Docker build Step nicht korrekt
# funktioniert, daher über SHELL Variable ausführen.
SHELL ["conda", "run", "-n", "uitb-sim2vr", "/bin/bash", "-c"]

# nötig? keine Ahnung
#RUN conda install -c conda-forge ffmpeg \
#        && conda clean --yes --all

RUN pip install -e .
RUN echo 'export MUJOCO_GL=egl' >> ~/.bashrc

RUN conda install -c conda-forge gcc=12.1.0

#RUN pip install -e third_party/pytorch3d \
#        && pip install -e third_party/kmeans_pytorch \
#        && python -m pip install detectron2 -f \
#                https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# restliche Configs/Skripte kopieren
#COPY . /banmo

# zum Testen interaktiv (docker run -it) in Bash rein
# Conda Environment automatisch aktivieren
#ENTRYPOINT ["/bin/bash"]

CMD [ "sleep", "infinity" ]