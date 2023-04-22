# IberLEF2023

All requirements were installed in a Python 3.11 environment.

First run the following command to install the requirements:

```bash
pip install -r reqs.txt
```

Furthermore, run the following command to install PyTorch for your GPU:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Change the ```pytorch-cuda``` argument according to your CUDA version. You can check your CUDA version with the following command:

```bash
nvcc --version
```

Then, to run the training script on a tiny dataset with the following command (Omit the ```tiny``` argument to use the full dataset):

```bash
python trainer.py -a 'cpu' -b 2 -n 2 -e 3 -c 'simple' -tiny 
```

Argument description:
* -a: accelerator (cpu or gpu)
* -b: batch size
* -n: number of workers
* -e: number of epochs
* -c: model architecture (simple for simple classifier)
* -tiny: activates use of tiny dataset (for testing purposes)
* -practise: activates use of practise dataset (subset of full dataset)

If none of the arguments are specified, the script will run with the following default values:
* -a: cpu
* -b: from yaml file
* -n: 2
* -e: from yaml file
* -c: simple
* -tiny: deactivated
* -practise: deactivated

