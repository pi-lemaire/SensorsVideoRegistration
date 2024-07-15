# Registering unmanned aerial vehicle videos in the long term

![](output_examples/roundabout.gif)
![](output_examples/mocopo.gif)
![](output_examples/360degrees.gif)

This is a python-only re-implementation of the paper:
[**Lemaire, Pierre, et al. "Registering unmanned aerial vehicle videos in the long term." Sensors 21.2 (2021): 513.**](https://www.mdpi.com/1424-8220/21/2/513)

Upon rewriting in python, the code has been slightly simplified compared to the original paper. In particular, 2 elements have been modified: a ratio has been applied to the registration correction ; the trajectory has been updated with the correction, instead of being fully independent.
The only package required are opencv and numpy.

An example on how to use the code is provided in the jupyter notebook, which uses ipywidgets too.

One low-res, short video example is provided in this repository. Other short-term examples are available here:
https://drive.google.com/drive/folders/1fxvRjgAkipjrGmuu4-8Oz5p6D4TfG7Qw?usp=sharing
