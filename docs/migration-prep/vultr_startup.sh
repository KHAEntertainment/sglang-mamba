#!/bin/bash
# Vultr Startup Script for SGLang-Mamba A100 VM

# 1. System Updates & Dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git python3-pip python3-venv curl wget

# 2. Setup Repository
cd /root
git clone https://github.com/KHAEntertainment/sglang-mamba.git
cd sglang-mamba
pip3 install --break-system-packages --upgrade pip
pip3 install --break-system-packages -e "python[all]"

# 3. Download the Granite Model (Pre-fetching to save time)
pip3 install --break-system-packages huggingface_hub
mkdir -p /mnt/models/granite-4.0-h-tiny
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ibm-granite/granite-4.0-tiny-mamba2-instruct',
                  local_dir='/mnt/models/granite-4.0-h-tiny')
"

# 4. Prepare directories for Claude / State restoration
mkdir -p /root/.remember
mkdir -p /root/.claude/projects/-root-sglang-mamba/memory/

echo "Vultr Startup Script Complete!" > /root/startup_finished.log
