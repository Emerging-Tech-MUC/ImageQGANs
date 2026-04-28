"""
FID vs. Shot Budget Sweep
=========================
Evaluates how FID changes as the number of measurement shots per image varies.
No retraining required: the decoder is swapped post-hoc on a pre-trained model.

Model: c9ece33a — trained with 2048 shots.

Aligned with FID_evaluation.ipynb:
  - Real data loaded from apps/logistics/training_data/ (local .npy)
  - Generated samples saved as .npy then converted to PNGs
  - Same pytorch_fid pipeline (save_fid_stats + calculate_fid_given_paths)
  - model_handler.sample() for generation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
from PIL import Image
from tqdm import tqdm
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3

from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler
from qugen.main.generator.measurements import decoder_factory

# ---------------------------------------------------------------------------
# Matplotlib style (consistent with plots_paper.ipynb)
# ---------------------------------------------------------------------------
for _font_file in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
    if 'Times' in _font_file:
        font_manager.fontManager.addfont(_font_file)
try:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
except Exception:
    pass

color_palette = {
    'blue': (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    'red': (0.8352941176470589, 0.3686274509803922, 0.0),
    'green': (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    'orange': (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
    'pink': (0.8,  0.47058823529411764,  0.7372549019607844), 
    'yellow': (0.9254901960784314, 0.8823529411764706, 0.2),
    'gray': (0.9, 0.9, 0.9)
    }

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
# Reduced sweep for fast initial run (3 points; expand later).
SHOTS_SWEEP = [256, 512, 1024, 2048, 4096, 8192, None]

MODEL_SHOTS_2048 = 'continuous_mnist_0_1_32x32_N_12665_minmax_qgan_c9ece33a'

EPOCH          = 15000
N_FID_SAMPLES  = 400    # small for fast initial run (~40s FID stats/folder); increase later
BATCH_SIZE     = 50     # same as FID_evaluation.ipynb
N_BOOTSTRAP    = 100    # bootstrap iterations for FID std estimation

# Local real-data .npy (already 32x32, values 0-255, no normalization needed)
REAL_DATA_NPY  = 'apps/logistics/training_data/mnist_0_1_32x32_N_12665.npy'

IMAGE_FOLDER_ROOT = 'images_FID/'
os.makedirs(IMAGE_FOLDER_ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers – mirrored from FID_evaluation.ipynb
# ---------------------------------------------------------------------------

def prepare_real_samples(image_folder, path_npy_samples, n_samples=N_FID_SAMPLES):
    """Save real MNIST 0&1 images (from local .npy) as uncompressed PNGs."""
    image_folder_path = os.path.join(IMAGE_FOLDER_ROOT, image_folder)
    if os.path.exists(image_folder_path):
        print(f'{image_folder}: image folder already exists, skipping.')
        return
    os.makedirs(image_folder_path)

    images = np.load(path_npy_samples)          # (N, 32, 32), float64 0-255
    np.random.seed(42)
    np.random.shuffle(images)
    images = images[:n_samples].clip(0, 255).astype(np.uint8)

    for i, img_arr in enumerate(images):
        Image.fromarray(img_arr, mode='L').save(
            os.path.join(image_folder_path, f'img_{i}.png'), compress_level=0
        )
    print(f'  -> Saved {len(images)} real images to {image_folder_path}')


def sample_model_to_npy(model_name, inference_shots, npy_path,
                         epoch=EPOCH, n_samples=N_FID_SAMPLES, batch_size=BATCH_SIZE):
    """
    Load pre-trained model, swap decoder to use inference_shots, generate
    n_samples images, save as .npy in [0,1] range.
    Mirrors reload_qgan_and_sample() in FID_evaluation.ipynb.
    """
    if os.path.exists(npy_path):
        print(f'  npy exists, skipping generation: {npy_path}')
        return

    model = ContinuousQGANModelHandler()
    model.reload(model_name=model_name, epoch=epoch)

    # Post-hoc decoder swap (same pattern as Figure 7 in plots_paper.ipynb)
    model.shots = inference_shots
    model.decoder = decoder_factory(
        model.decoding_scheme,
        n_ancilla_qubits=model.n_ancilla_qubits,
        shots=inference_shots,
    )

    all_samples = np.empty((n_samples, 32 * 32))
    n_collected = 0
    pbar = tqdm(total=n_samples, desc=f'  generating shots={inference_shots}')
    while n_collected < n_samples:
        n_batch = min(batch_size, n_samples - n_collected)
        batch = model.sample(n_batch)           # [0, 1] range, shape (n_batch, 1024)
        all_samples[n_collected:n_collected + n_batch] = batch.reshape(n_batch, -1)
        n_collected += n_batch
        pbar.update(n_batch)
    pbar.close()

    np.save(npy_path, all_samples)
    print(f'  -> Saved {n_collected} samples to {npy_path}')


def prepare_generated_samples(image_folder, npy_path, n_samples=N_FID_SAMPLES):
    """
    Convert generated .npy samples ([0,1] range) to uncompressed PNGs.
    Mirrors the prepare_samples() conversion in FID_evaluation.ipynb.
    """
    image_folder_path = os.path.join(IMAGE_FOLDER_ROOT, image_folder)
    if os.path.exists(image_folder_path):
        print(f'{image_folder}: image folder already exists, skipping.')
        return
    os.makedirs(image_folder_path)

    images = np.load(npy_path)                  # (N, 1024), float in [0, 1]
    np.random.seed(42)
    np.random.shuffle(images)
    images = images[:n_samples]
    images = (255 * images).clip(0, 255).astype(np.uint8).reshape(-1, 32, 32)

    for i, img_arr in enumerate(images):
        Image.fromarray(img_arr, mode='L').save(
            os.path.join(image_folder_path, f'img_{i}.png'), compress_level=0
        )
    print(f'  -> Wrote {len(images)} PNGs to {image_folder_path}')


def compute_fid_stats(image_folder):
    """Compute and cache InceptionV3 activation statistics. Mirrors FID_evaluation.ipynb."""
    stats_path = os.path.join(IMAGE_FOLDER_ROOT, f'{image_folder}.npz')
    if os.path.exists(stats_path):
        print(f'{image_folder}: FID stats already exist, skipping.')
        return
    image_folder_path = os.path.join(IMAGE_FOLDER_ROOT, image_folder)
    fid_score.save_fid_stats(
        paths=[image_folder_path, stats_path],
        batch_size=50, device=None, dims=2048,
        num_workers=0,
    )
    print(f'  -> Saved FID stats to {stats_path}')


def get_fid(gen_folder, real_folder):
    """Compute FID from pre-computed stats. Mirrors FID_evaluation.ipynb."""
    gen_stats  = os.path.join(IMAGE_FOLDER_ROOT, gen_folder  + '.npz')
    real_stats = os.path.join(IMAGE_FOLDER_ROOT, real_folder + '.npz')
    return fid_score.calculate_fid_given_paths(
        paths=[gen_stats, real_stats],
        batch_size=50, device=None, dims=2048,
    )


def get_activations_cached(image_folder, dims=2048):
    """Extract InceptionV3 activations for a folder, cached to disk as _acts.npy."""
    acts_path = os.path.join(IMAGE_FOLDER_ROOT, image_folder + '_acts.npy')
    if os.path.exists(acts_path):
        return np.load(acts_path)
    image_folder_path = os.path.join(IMAGE_FOLDER_ROOT, image_folder)
    files = sorted(
        os.path.join(image_folder_path, f)
        for f in os.listdir(image_folder_path) if f.endswith('.png')
    )
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.eval()
    acts = fid_score.get_activations(files, model, batch_size=50, dims=dims,
                                     device='cpu', num_workers=0)
    np.save(acts_path, acts)
    return acts


def get_fid_with_std(gen_folder, real_folder, n_bootstrap=N_BOOTSTRAP):
    """Compute FID mean +/- std via bootstrap resampling of InceptionV3 activations."""
    gen_acts  = get_activations_cached(gen_folder)
    real_acts = get_activations_cached(real_folder)
    rng = np.random.default_rng(42)
    fids = []
    for _ in range(n_bootstrap):
        g = gen_acts[rng.choice(len(gen_acts),   len(gen_acts),   replace=True)]
        r = real_acts[rng.choice(len(real_acts), len(real_acts), replace=True)]
        try:
            fids.append(fid_score.calculate_frechet_distance(
                g.mean(0), np.cov(g, rowvar=False),
                r.mean(0), np.cov(r, rowvar=False),
            ))
        except Exception:
            pass  # skip singular matrices in degenerate resamples
    return float(np.mean(fids)), float(np.std(fids))


if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # 1. Real data statistics
    # -----------------------------------------------------------------------
    REAL_FOLDER = 'real_MNIST_0_1'
    print('=== Preparing real data ===')
    prepare_real_samples(REAL_FOLDER, REAL_DATA_NPY)
    compute_fid_stats(REAL_FOLDER)

    # -----------------------------------------------------------------------
    # 2. Sweep: generate + FID for each shots count
    # -----------------------------------------------------------------------
    fid_shots_model = {}

    print('\n=== Model: shots_2048 ===')
    for shots in SHOTS_SWEEP:
        label    = str(shots) if shots is not None else 'exact'
        folder   = f'shots_sweep_shots_2048_{label}'
        npy_path = os.path.join(IMAGE_FOLDER_ROOT, f'{folder}.npy')

        print(f'\n[shots_2048] inference shots = {label}')
        sample_model_to_npy(MODEL_SHOTS_2048, shots, npy_path)
        prepare_generated_samples(folder, npy_path)
        compute_fid_stats(folder)
        fid_val, fid_std = get_fid_with_std(folder, REAL_FOLDER)
        fid_shots_model[shots] = (fid_val, fid_std)
        print(f'  -> FID = {fid_val:.2f} +/- {fid_std:.2f}')

    # -----------------------------------------------------------------------
    # 3. Results table
    # -----------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('FID vs. Shot Budget — MNIST digits 0 & 1 (32 x 32)')
    print(f'N_FID_SAMPLES = {N_FID_SAMPLES}')
    col = '{:>18}  {:>28}'
    print(col.format('Shots/image', 'FID (shots-trained)'))
    print('-' * 48)
    for shots in SHOTS_SWEEP:
        label = str(shots) if shots is not None else 'inf (SV exact)'
        sm_mean, sm_std = fid_shots_model[shots]
        print(col.format(label, f'{sm_mean:.2f} +/- {sm_std:.2f}'))

    # -----------------------------------------------------------------------
    # 4. Plot
    # -----------------------------------------------------------------------
    finite_shots  = [s for s in SHOTS_SWEEP if s is not None]
    fid_sm_finite = [fid_shots_model[s][0] for s in finite_shots]
    fid_sm_std    = [fid_shots_model[s][1] for s in finite_shots]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.errorbar(finite_shots, fid_sm_finite, yerr=fid_sm_std,
                marker='s', linewidth=1.5, markersize=5, capsize=3,
                color=color_palette['red'],
                label='Shots-trained (2048, inference shots varied)')
    if None in fid_shots_model:
        ax.axhline(fid_shots_model[None][0], linestyle='--', linewidth=1.5,
                   color=color_palette['red'], alpha=0.55,
                   label='Shots-trained + SV inference')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Inference shots per image', fontsize=11)
    ax.set_ylabel('FID', fontsize=11)
    ax.set_title('FID vs. Shot Budget — MNIST digits 0 & 1 (32 x 32)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fid_vs_shots.pdf', bbox_inches='tight')
    plt.show()
