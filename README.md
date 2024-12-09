# ClinicalBertSnomed
This is some example code for using ClinicalBert to suggest SNOMED codes for medical documents.
ClinicalBert is a large language model trained on clinical data and available as open source from Hugging Face.

ClinicalBert is installed using Anaconda and an environment setup with Python 3.8 installed. This is pre-requisite for running this code.
Python environment must be set to use the one created in Anaconda
Note GPU with CUDA cores will run way better than running this on the CPU, but it will run on the CPU. Check CUDA version in NVidia Control Panel.

Install Anaconda and run Anaconda Terminal (MacOS)/Powershell (Windows) as Admin

conda create --name clinicalbert-env python=3.8 (Important - anything above Python 3.8 not supported by pytorch below)

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip3 install transformers

Can be run directly from VS Code by setting the python environment to clinicalbert-env from Anaconda
