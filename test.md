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
