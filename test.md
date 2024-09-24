# Distributed GPU Training on Jetstream 2 Cloud

This guide documents the steps taken to set up two GPU instances on Jetstream 2 Cloud and run a distributed GPU training job using PyTorch’s `torchrun`. The process involves setting up SSH access, configuring the OpenStack environment, installing required software, and running the training script across two GPU nodes.

---

## **Prerequisites**

- Access to Jetstream 2 Cloud instances
- Basic knowledge of Linux commands and SSH
- Python installed with `torch` and `torchvision` libraries

---

## **Step 1: Start Instances and Access Details**

1. Create two instances on Jetstream 2 Cloud:
    - **Instance 1**: `parallelism-test 1 of 2`
    - **Instance 2**: `parallelism-test 2 of 2`
    
    Note the **public IP addresses** for both instances.

2. Obtain the following credentials from the Jetstream 2 dashboard:
   - Username: `exouser`
   - SSH Public Key Name
   - Passphrase for SSH Key

---

## **Step 2: Configure SSH Access Between Instances**

1. SSH into **Instance 1** using the credentials provided:
   ```bash
   ssh exouser@<Public_IP_Instance_1>
Generate SSH keys on Instance 1 (if not already generated):

bash
Copy code
ssh-keygen -t rsa -b 4096
Copy the public key to Instance 2:

bash
Copy code
ssh-copy-id -i ~/.ssh/id_rsa.pub exouser@<Public_IP_Instance_2>
Verify SSH access from Instance 1 to Instance 2:

bash
Copy code
ssh exouser@<Public_IP_Instance_2>
You should be able to connect without a password prompt.

Step 3: Set Up OpenStack Environment
Download the openrc.sh file from the Jetstream 2 dashboard and transfer it to Instance 1 and Instance 2.

On each instance, source the openrc.sh file to set up the OpenStack environment:

bash
Copy code
source /path/to/CIS240139_IU-openrc.sh
Step 4: Install NVIDIA Drivers and CUDA Toolkit
Install NVIDIA drivers and CUDA Toolkit on both instances. For Ubuntu, you can use:

bash
Copy code
sudo apt-get update
sudo apt-get install nvidia-driver-535 nvidia-cuda-toolkit -y
Verify that the drivers are installed correctly:

bash
Copy code
nvidia-smi
Step 5: Install PyTorch and Other Required Libraries
Install PyTorch and torchvision:

bash
Copy code
pip install torch torchvision --upgrade
Verify PyTorch and CUDA installation:

python
Copy code
python3 -c "import torch; print(torch.cuda.is_available())"
Step 6: Prepare Training Script
Upload your train_script.py to both instances. Make sure it includes the necessary PyTorch torch.distributed imports for distributed training.

Example import statement:

python
Copy code
from torch.nn.parallel import DistributedDataParallel as DDP
Step 7: Set Up Environment Variables for Distributed Training
On Instance 1 (Master Node), set the following environment variables:

bash
Copy code
export MASTER_ADDR="<Public_IP_Instance_1>"
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0
On Instance 2, set the environment variables:

bash
Copy code
export MASTER_ADDR="<Public_IP_Instance_1>"
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=1
Step 8: Running the Distributed Training Script
Run the training script on both instances using torchrun.

On Instance 1:

bash
Copy code
torchrun --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_script.py
On Instance 2:

bash
Copy code
torchrun --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_script.py
Step 9: Debugging Issues with NCCL
If you encounter issues, set the following environment variables on both instances:

bash
Copy code
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
Verify the training process using nvidia-smi:

bash
Copy code
nvidia-smi
You should see GPU usage as the training script runs.

Conclusion
You have successfully set up a distributed training environment using two GPU instances on Jetstream 2 Cloud. This setup allows for efficient parallelism and distributed deep learning model training.

