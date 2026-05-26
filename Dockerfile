FROM nvcr.io/nvidia/pytorch:22.09-py3

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-venv \
    libstdc++6 \
    libgcc-s1 \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Set system Python as default
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

COPY requirements.txt /workspace/codes/requirements.txt

# Upgrade pip and install dependencies with the right CUDA wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /workspace/codes/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

CMD ["/bin/bash"]
