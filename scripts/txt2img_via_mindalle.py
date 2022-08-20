"""
Go txt2img using DALLE-Mini Mega (via min-dalle) for txt2img, then Stable Diffusion for img2img.

Inspired by HardMaru's tweet: https://twitter.com/hardmaru/status/1559330378426753024

On top of the base Stable Diffusion setup, requires a simple:
  `pip install min-dalle`
"""

# min-dalle imports
import argparse
import os
from PIL import Image
from min_dalle import MinDalle
import torch

# sd imports
import argparse, os, sys, glob
import PIL
#import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from img2img import chunk, load_model_from_config, load_img


def save_grid(image_paths, grid_path, size=256):
    """Combine multiple images into a grid
    https://stackoverflow.com/a/10649311
    """
    # NOTE: assumes each individual image is 256x256
    h_w = int(len(image_paths)**0.5)
    assert h_w**2 == len(image_paths), f"Number of images {len(image_paths)} is not square"

    grid_im = Image.new('RGB', (size*h_w,size*h_w))
    idx = 0
    for i in range(0,size*h_w,size):
        for j in range(0,size*h_w,size):
            im = Image.open(image_paths[idx])
            grid_im.paste(im, (j, i))  # flip h,w to get same orientation as source grid
            idx += 1

    if grid_path:
        grid_im.save(grid_path)


def split_grid(path: str):
    """Turns a NxN grid of 256x256 images (from min-dalle) into separate PIL image objects"""
    grid_image = Image.open(path)

    # NOTE: assumes each individual image is 256x256
    size = 256
    n_h = grid_image.height // size
    n_w = grid_image.width // size
    assert n_h == n_w, (n_h, n_w)
    
    split_images = []
    for h in range(n_h):
        for w in range(n_w):
            # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
            # (left, upper, right, lower)
            # [h*size:(h+1)*size, w*size:(w+1)*size]
            split_image = grid_image.crop((
                w*size, # left
                h*size,  # upper
                (w+1)*size,  # right
                (h+1)*size,  # lower
            ))
            split_images.append(split_image)
    
    return split_images


def save_image(image: Image.Image, path: str):
    if os.path.isdir(path):
        path = os.path.join(path, 'generated.png')
    elif not path.endswith('.png'):
        path += '.png'
    print("saving image to", path)
    image.save(path)
    return image


def mindalle_generate_image(
    is_mega: bool,
    text: str,
    seed: int,
    grid_size: int,
    top_k: int,
    image_path: str,
    models_root: str,
    fp16: bool,
):
    """https://github.com/kuprel/min-dalle/blob/main/image_from_text.py#L38-L64"""
    model = MinDalle(
        is_mega=is_mega, 
        models_root=models_root,
        is_reusable=False,
        is_verbose=True,
        dtype=torch.float16 if fp16 else torch.float32
    )

    image = model.generate_image(
        text, 
        seed, 
        grid_size, 
        top_k=top_k, 
        is_verbose=True
    )
    save_image(image, image_path)
    #print(ascii_from_image(image, size=128))

    print("Flushing model...")
    del model


def sd_img2img(opt):
    """https://github.com/cpacker/stable-diffusion/blob/main/scripts/img2img.py#L60-L289"""

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    output_image_paths = []  # NOTE new
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                im_out = os.path.join(sample_path, f"{base_count:05}.png")
                                output_image_paths.append(im_out)  # NOTE new
                                Image.fromarray(x_sample.astype(np.uint8)).save(im_out)
                                base_count += 1
                        all_samples.append(x_samples)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n{output_image_paths}")
    return output_image_paths


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mindalle-upscale', action='store_true')
    parser.set_defaults(mindalle_upscale=True)

    ###################################################
    ########      min-dalle (txt2img) args     ########
    ###################################################

    # See: https://github.com/kuprel/min-dalle/blob/main/image_from_text.py
    # TODO expose all args from model.generate_image(...)
    parser.add_argument('--mega', action='store_true')
    parser.add_argument('--no-mega', dest='mega', action='store_false')
    parser.set_defaults(mega=True)  # NOTE: use mega by default
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--text', type=str, default='Dali painting of WALL·E')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--grid-size', type=int, default=1)
    #parser.add_argument('--image-path', type=str, default='generated')
    parser.add_argument('--models-root', type=str, default='pretrained')
    parser.add_argument('--top_k', type=int, default=256)

    ###################################################
    ########         SD img2img args           ########
    ###################################################

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        #default="a painting of a virus monster playing guitar",
        default=None,
        help="the prompt to render"
    )
    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        #default="outputs/img2img-samples"
        default="outputs/mindalle-txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        #default=2,
        default=1,  # NOTE: diff
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--strength",
        type=float,
        #default=0.75,
        default=0.1,  # NOTE diff
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--sd_seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    args = parser.parse_args()

    ### First generate the base images with min-DALLE

    # Store the min-dalle image in the same directory as SD images
    os.makedirs(args.outdir, exist_ok=True)
    outpath = args.outdir
    
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    #grid_count = len(os.listdir(outpath)) - 1

    image_path = os.path.join(sample_path, f"{base_count:05}_mindalle.png")

    min_dalle_out = mindalle_generate_image(
        is_mega=args.mega,
        text=args.text,
        seed=args.seed,
        grid_size=args.grid_size,
        top_k=args.top_k,
        image_path=image_path,
        models_root=args.models_root,
        fp16=args.fp16,
    )

    ### Then run Stable Diffusion img2img with the min-DALLE results as the seed image

    # By default, pass the min-dalle prompt through the SD
    if args.prompt is None:
        args.prompt = args.text
        print(f"Set SD prompt to {args.prompt}")
    else:
        print("Warning, separate prompt specified for SD!")
        print(f"Set SD prompt to {args.prompt}")

    if args.grid_size > 1:
        # If we ran min-dalle in grid format:
        # (1) split the grid into individual images
        mindalle_images = split_grid(image_path)
        sd_images = []
        # (2) run each image in the grid through SD serially
        for i, img in enumerate(mindalle_images):
            # Write out the split off min-dalle image
            base_count = len(os.listdir(sample_path))
            img_path = os.path.join(sample_path, f"{base_count:05}_mindalle_s{i}.png")
            # NOTE: SD seems very sensitive to the input resolution, so upscale (naively) the min-dalle image to SD native res
            if args.mindalle_upscale:
                # Does bicubic by default
                print(f"Upscaling {img_path} to SD native res...")
                img = img.resize((512,512))
            img.save(img_path)
            # Feed it through SD
            args.init_img = img_path
            sd_im_path = sd_img2img(args)
            if len(sd_im_path) > 1:
                print(f"Warning: Stable Diffusion generated {len(sd_im_path)} samples, only using the first one for the grid")
            sd_images.append(sd_im_path[0])

        # Save a grid of the SD outputs in the same order as the min-dalle grid
        # (so that we can do a side-by-side)
        base_count = len(os.listdir(sample_path))
        sd_grid_path = os.path.join(sample_path, f"{base_count:05}_sd.png")
        save_grid(sd_images, sd_grid_path, size=(512 if args.mindalle_upscale else 256))

    else:
        args.init_img = image_path
        sd_img2img(args)

if __name__ == "__main__":
    main()