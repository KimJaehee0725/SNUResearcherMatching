FROM nvcr.io/nvidia/pytorch:24.08-py3

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

COPY demo/requirements.txt /workspace/codes/demo/requirements.txt

# Upgrade pip and install only demo runtime dependencies.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /workspace/codes/demo/requirements.txt

CMD ["/bin/bash"]
