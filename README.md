# IberLEF2023

All requirements were installed in a Python 3.8 environment.

Run with

```python

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