import numpy as np
from PIL import Image
from scipy.stats import skew, kurtosis, entropy


def compute_entropy(feature_map, global_min, global_max, base=2):
    flat = feature_map.flatten()
    num_bins = flat.size
    hist, _ = np.histogram(flat, bins=num_bins, range=(global_min, global_max), density=True)
    return entropy(hist, base=base)

def compute_rgb_joint_entropy(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # 各チャンネルをフラット化して、(N, 3) の配列に
    pixels = img_array.reshape(-1, 3)
    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    # 3次元ヒストグラムを計算
    hist, _ = np.histogramdd((r, g, b), bins=(256, 256, 256), range=((0, 256), (0, 256), (0, 256)), density=True)

    # 非ゼロの確率成分だけ取り出してエントロピー計算
    hist_nonzero = hist[hist > 0]
    joint_entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

    return joint_entropy

def calculate_entropy(img_channel, bins=256, value_range=(0, 256)):
    hist, _ = np.histogram(img_channel, bins=bins, range=value_range, density=True)
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    return entropy

def get_hsv_entropies(image_path):
    img = Image.open(image_path).convert('HSV')
    img_array = np.array(img)
    
    # HSV
    H_channel = img_array[:, :, 0]
    S_channel = img_array[:, :, 1]
    V_channel = img_array[:, :, 2]
    
    h_entropy = calculate_entropy(H_channel, bins=256, value_range=(0, 256))
    s_entropy = calculate_entropy(S_channel, bins=256, value_range=(0, 256))
    v_entropy = calculate_entropy(V_channel, bins=256, value_range=(0, 256))
    
    return (h_entropy, s_entropy, v_entropy)


def get_ycbcr_entropies(image_path):
    img = Image.open(image_path).convert('YCbCr')
    img_array = np.array(img)
    
    Y_channel = img_array[:, :, 0]
    Cb_channel = img_array[:, :, 1]
    Cr_channel = img_array[:, :, 2]
    
    y_entropy = calculate_entropy(Y_channel, bins=256, value_range=(0, 256))
    cb_entropy = calculate_entropy(Cb_channel, bins=256, value_range=(0, 256))
    cr_entropy = calculate_entropy(Cr_channel, bins=256, value_range=(0, 256))
    
    return (y_entropy, cb_entropy, cr_entropy)


