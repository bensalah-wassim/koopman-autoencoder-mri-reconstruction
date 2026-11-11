# External modules
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse


def compute_metric(metric, gt_img, image):
    # Compute the corresponding metric
    if metric == "ssim":
        metric_value = ssim(gt_img, image,
                            data_range=gt_img.max()-gt_img.min())
    elif metric == "mse":
        metric_value = mse(gt_img, image)
    elif metric == "psnr":
        metric_value = psnr(gt_img, image)
    else:
        metric_value = nrmse(gt_img, image)
    return metric_value