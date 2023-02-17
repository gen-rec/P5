FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install dependencies
RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge && \
    pip3 install transformers==4.2.1 sentencepiece pyyaml chardet

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Run bash when the container launches
WORKDIR /workspace
CMD ["/bin/bash"]
