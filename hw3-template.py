from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    raise NotImplementedError

def get_covariance(dataset):
    # Your implementation goes here!
    raise NotImplementedError

def get_eig(S, k):
    # Your implementation goes here!
    raise NotImplementedError

def get_eig_prop(S, prop):
    # Your implementation goes here!
    raise NotImplementedError

def project_and_reconstruct_image(image, U):
    # Your implementation goes here!
    raise NotImplementedError

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    fig.tight_layout()

    # Your implementation goes here!

    return fig, ax1, ax2, ax3

def perturb_image(image, U, sigma):
    # Your implementation goes here!
    raise NotImplementedError