from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    x_mean = np.mean(x, axis=0)
    x_center = x-x_mean
    return x_center

    raise NotImplementedError

def get_covariance(dataset):
    transpose_x=np.transpose(dataset)
    n=dataset.shape[0]
    dot_product=np.dot(transpose_x,dataset)
    covariance_x= (dot_product)*(1/(n-1))
    return covariance_x
    raise NotImplementedError

def get_eig(S, k):
    # Your implementation goes here!
    d = S.shape[0]
    eigvals, eigvecs = eigh(S, subset_by_index=[d - k, d - 1])
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    Lambda = np.diag(eigvals)

    return Lambda, eigvecs
    raise NotImplementedError

def get_eig_prop(S, prop):
    # Your implementation goes here!
    eigvals, eigvecs = eigh(S)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    total_variance = np.sum(eigvals)
    variance_ratio = eigvals / total_variance
    selected_indices = np.where(variance_ratio > prop)[0]
    selected_eigvals = eigvals[selected_indices]
    selected_eigvecs = eigvecs[:, selected_indices]
    Lambda = np.diag(selected_eigvals)

    return Lambda, selected_eigvecs
    raise NotImplementedError

def project_and_reconstruct_image(image, U):
    # Your implementation goes here!
    alpha = np.dot(U.T, image)

    x_reconstructed = np.dot(U, alpha)

    return x_reconstructed
    raise NotImplementedError

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)

    # Your implementation goes here!
    im_orig_fullres_reshaped = im_orig_fullres.reshape(218, 178, 3)
    ax1.imshow(im_orig_fullres_reshaped)
    ax1.set_title('Original High Res')
    ax1.set_aspect('equal')

    im_orig_reshaped = im_orig.reshape(60, 50)
    img2 = ax2.imshow(im_orig_reshaped, aspect='equal', cmap='gray')
    ax2.set_title('Original')
    fig.colorbar(img2, ax=ax2)

    im_reconstructed_reshaped = im_reconstructed.reshape(60, 50)
    img3 = ax3.imshow(im_reconstructed_reshaped, aspect='equal', cmap='gray') 
    ax3.set_title('Reconstructed')
    fig.colorbar(img3, ax=ax3)

    return fig, ax1, ax2, ax3

def perturb_image(image, U, sigma):
    # # Your implementation goes here!
    alpha = np.dot(U.T, image)
    noise = np.random.normal(0, sigma, alpha.shape)
    perturbed_alpha = alpha + noise
    x_perturbed = np.dot(U, perturbed_alpha)
    return x_perturbed
    raise NotImplementedError


# X = load_and_center_dataset('celeba_60x50.npy')
# S = get_covariance(X)
# Lambda, U = get_eig(S, 50)
# celeb_idx = 34
# x = X[celeb_idx]
# x_fullres = np.load('celeba_218x178x3.npy')[celeb_idx]
# reconstructed = project_and_reconstruct_image(x, U)
# fig, ax1, ax2, ax3 = display_image(x_fullres, x, reconstructed)
# plt.show()

# X = load_and_center_dataset('celeba_60x50.npy')
# S = get_covariance(X)
# Lambda, U = get_eig(S, 50)
# celeb_idx = 34
# x = X[celeb_idx]
# x_fullres = np.load('celeba_218x178x3.npy')[celeb_idx]
# x_perturbed = perturb_image(x, U, sigma=1000)
# fig, ax1, ax2, ax3 = display_image(x_fullres, x, x_perturbed)
# plt.show()show