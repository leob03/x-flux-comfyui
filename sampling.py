import math
from typing import Callable, List

import torch
from einops import rearrange, repeat
from torch import Tensor
import numpy as np

#from .modules.conditioner import HFEmbedder
from .layers import DoubleStreamMixerProcessor, timestep_embedding
from tqdm.auto import tqdm
from .utils import ControlNetContainer
def model_forward(
    model,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    block_controlnet_hidden_states=None,
    guidance: Tensor | None = None,
    neg_mode: bool | None = False,
) -> Tensor:
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")
    # running on sequences img
    img = model.img_in(img)
    vec = model.time_in(timestep_embedding(timesteps, 256))
    if model.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + model.guidance_in(timestep_embedding(guidance, 256))
    vec = vec + model.vector_in(y)
    txt = model.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = model.pe_embedder(ids)
    if block_controlnet_hidden_states is not None:
        controlnet_depth = len(block_controlnet_hidden_states)
    for index_block, block in enumerate(model.double_blocks):
        if hasattr(block, "processor"):
            if isinstance(block.processor, DoubleStreamMixerProcessor):
                if neg_mode:
                    for ip in block.processor.ip_adapters:
                        ip.ip_hidden_states = ip.in_hidden_states_neg
                else:
                    for ip in block.processor.ip_adapters:
                        ip.ip_hidden_states = ip.in_hidden_states_pos

        img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        # controlnet residual

        if block_controlnet_hidden_states is not None:
            img = img + block_controlnet_hidden_states[index_block % 2]


    img = torch.cat((txt, img), 1)
    for block in model.single_blocks:
        img = block(img, vec=vec, pe=pe)
    img = img[:, txt.shape[1] :, ...]

    img = model.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(txt_t5, vec_clip, img: Tensor) -> dict[str, Tensor]:
    txt = txt_t5
    vec = vec_clip
    bs, c, h, w = img.shape


    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)


    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)

    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device, dtype=img.dtype),
        "txt": txt.to(img.device, dtype=img.dtype),
        "txt_ids": txt_ids.to(img.device, dtype=img.dtype),
        "vec": vec.to(img.device, dtype=img.dtype),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


## Multi-step
def denoise_multistep(
    model,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs=1,
    timestep_to_start_cfg=0,
    image2image_strength=None,
    orig_image=None,
    callback=None,
    width=512,
    height=512,
):
    i = 0

    if image2image_strength is not None and orig_image is not None:
        t_idx = int((1 - np.clip(image2image_strength, 0.0, 1.0)) * len(timesteps))
        t = timesteps[t_idx]
        timesteps = timesteps[t_idx:]
        orig_image = rearrange(orig_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).to(img.device, dtype=img.dtype)
        img = t * img + (1.0 - t) * orig_image

    img_ids = img_ids.to(img.device, dtype=img.dtype)
    txt = txt.to(img.device, dtype=img.dtype)
    txt_ids = txt_ids.to(img.device, dtype=img.dtype)
    vec = vec.to(img.device, dtype=img.dtype)

    if hasattr(model, "guidance_in"):
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    else:
        guidance_vec = None  # Ignored for models without guidance

    total_steps = len(timesteps) - 1
    pred_n_minus1 = None  # To store v_{n-1}
    pred_n = None         # To store v_n

    for idx, (t_curr, t_prev) in enumerate(tqdm(zip(timesteps[:-1], timesteps[1:]), desc="Sampling", total=total_steps)):
        h = t_prev - t_curr  # Time step size (negative for reverse time)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        if idx == 0:
            # First step: use RK4 to initialize
            # --- RK4 Implementation ---
            # k1
            pred1 = model_forward(
                model,
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )
            if i >= timestep_to_start_cfg:
                neg_pred1 = model_forward(
                    model,
                    img=img,
                    img_ids=img_ids,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    y=neg_vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    neg_mode=True,
                )
                pred1 = neg_pred1 + true_gs * (pred1 - neg_pred1)

            # k2
            img_k2 = img + (h / 2) * pred1
            t_mid = t_curr + (h / 2)
            t_mid_vec = torch.full((img.shape[0],), t_mid, dtype=img.dtype, device=img.device)
            pred2 = model_forward(
                model,
                img=img_k2,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_mid_vec,
                guidance=guidance_vec,
            )
            if i >= timestep_to_start_cfg:
                neg_pred2 = model_forward(
                    model,
                    img=img_k2,
                    img_ids=img_ids,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    y=neg_vec,
                    timesteps=t_mid_vec,
                    guidance=guidance_vec,
                    neg_mode=True,
                )
                pred2 = neg_pred2 + true_gs * (pred2 - neg_pred2)

            # k3
            img_k3 = img + (h / 2) * pred2
            pred3 = model_forward(
                model,
                img=img_k3,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_mid_vec,
                guidance=guidance_vec,
            )
            if i >= timestep_to_start_cfg:
                neg_pred3 = model_forward(
                    model,
                    img=img_k3,
                    img_ids=img_ids,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    y=neg_vec,
                    timesteps=t_mid_vec,
                    guidance=guidance_vec,
                    neg_mode=True,
                )
                pred3 = neg_pred3 + true_gs * (pred3 - neg_pred3)

            # k4
            img_k4 = img + h * pred3
            t_prev_vec = torch.full((img.shape[0],), t_prev, dtype=img.dtype, device=img.device)
            pred4 = model_forward(
                model,
                img=img_k4,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_prev_vec,
                guidance=guidance_vec,
            )
            if i >= timestep_to_start_cfg:
                neg_pred4 = model_forward(
                    model,
                    img=img_k4,
                    img_ids=img_ids,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    y=neg_vec,
                    timesteps=t_prev_vec,
                    guidance=guidance_vec,
                    neg_mode=True,
                )
                pred4 = neg_pred4 + true_gs * (pred4 - neg_pred4)

            # Update img
            img = img + (h / 6) * (pred1 + 2 * pred2 + 2 * pred3 + pred4)

            # Initialize velocities for multistep method
            pred_n_minus1 = pred1  # v_{n-1}
            pred_n = pred4         # v_n

        else:
            # Predictor: 2-step Adams-Bashforth
            img_predict = img + h * (1.5 * pred_n - 0.5 * pred_n_minus1)

            # Compute v_{n+1}^{predict}
            t_prev_vec = torch.full((img.shape[0],), t_prev, dtype=img.dtype, device=img.device)
            pred_np1_predict = model_forward(
                model,
                img=img_predict,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_prev_vec,
                guidance=guidance_vec,
            )
            if i >= timestep_to_start_cfg:
                neg_pred_np1_predict = model_forward(
                    model,
                    img=img_predict,
                    img_ids=img_ids,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    y=neg_vec,
                    timesteps=t_prev_vec,
                    guidance=guidance_vec,
                    neg_mode=True,
                )
                pred_np1_predict = neg_pred_np1_predict + true_gs * (pred_np1_predict - neg_pred_np1_predict)

            # Corrector: 2-step Adams-Moulton
            img = img + h * (0.5 * pred_np1_predict + 0.5 * pred_n)

            # Update velocities for next step
            pred_n_minus1 = pred_n
            pred_n = pred_np1_predict

        if callback is not None:
            unpacked = unpack(img.float(), height, width)
            callback(step=i, x=img, x0=unpacked, total_steps=total_steps)
        i += 1

    return img

## RK4
def denoise_rk4(
    model,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs=1,
    timestep_to_start_cfg=0,
    image2image_strength=None,
    orig_image=None,
    callback=None,
    width=512,
    height=512,
):
    i = 0

    if image2image_strength is not None and orig_image is not None:
        t_idx = int((1 - np.clip(image2image_strength, 0.0, 1.0)) * len(timesteps))
        t = timesteps[t_idx]
        timesteps = timesteps[t_idx:]
        orig_image = rearrange(orig_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).to(img.device, dtype=img.dtype)
        img = t * img + (1.0 - t) * orig_image

    img_ids = img_ids.to(img.device, dtype=img.dtype)
    txt = txt.to(img.device, dtype=img.dtype)
    txt_ids = txt_ids.to(img.device, dtype=img.dtype)
    vec = vec.to(img.device, dtype=img.dtype)

    if hasattr(model, "guidance_in"):
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    else:
        guidance_vec = None  # Ignored for models without guidance

    total_steps = len(timesteps) - 1
    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), desc="Sampling", total=total_steps):
        h = t_prev - t_curr  # Time step size (negative for reverse time)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        # k1
        pred1 = model_forward(
            model,
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        if i >= timestep_to_start_cfg:
            neg_pred1 = model_forward(
                model,
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                neg_mode=True,
            )
            pred1 = neg_pred1 + true_gs * (pred1 - neg_pred1)

        # k2
        img_k2 = img + (h / 2) * pred1
        t_mid1 = t_curr + (h / 2)
        t_mid1_vec = torch.full((img.shape[0],), t_mid1, dtype=img.dtype, device=img.device)
        pred2 = model_forward(
            model,
            img=img_k2,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_mid1_vec,
            guidance=guidance_vec,
        )
        if i >= timestep_to_start_cfg:
            neg_pred2 = model_forward(
                model,
                img=img_k2,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_mid1_vec,
                guidance=guidance_vec,
                neg_mode=True,
            )
            pred2 = neg_pred2 + true_gs * (pred2 - neg_pred2)

        # k3
        img_k3 = img + (h / 2) * pred2
        t_mid2 = t_curr + (h / 2)
        t_mid2_vec = torch.full((img.shape[0],), t_mid2, dtype=img.dtype, device=img.device)
        pred3 = model_forward(
            model,
            img=img_k3,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_mid2_vec,
            guidance=guidance_vec,
        )
        if i >= timestep_to_start_cfg:
            neg_pred3 = model_forward(
                model,
                img=img_k3,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_mid2_vec,
                guidance=guidance_vec,
                neg_mode=True,
            )
            pred3 = neg_pred3 + true_gs * (pred3 - neg_pred3)

        # k4
        img_k4 = img + h * pred3
        t_prev_vec = torch.full((img.shape[0],), t_prev, dtype=img.dtype, device=img.device)
        pred4 = model_forward(
            model,
            img=img_k4,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_prev_vec,
            guidance=guidance_vec,
        )
        if i >= timestep_to_start_cfg:
            neg_pred4 = model_forward(
                model,
                img=img_k4,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_prev_vec,
                guidance=guidance_vec,
                neg_mode=True,
            )
            pred4 = neg_pred4 + true_gs * (pred4 - neg_pred4)

        # Combine slopes to update img
        img = img + (h / 6) * (pred1 + 2 * pred2 + 2 * pred3 + pred4)

        if callback is not None:
            unpacked = unpack(img.float(), height, width)
            callback(step=i, x=img, x0=unpacked, total_steps=total_steps)
        i += 1

    return img

## RK2-Heun's method
def denoise_rk2(
    model,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs=1,
    timestep_to_start_cfg=0,
    image2image_strength=None,
    orig_image=None,
    callback=None,
    width=512,
    height=512,
):
    i = 0

    if image2image_strength is not None and orig_image is not None:
        t_idx = int((1 - np.clip(image2image_strength, 0.0, 1.0)) * len(timesteps))
        t = timesteps[t_idx]
        timesteps = timesteps[t_idx:]
        orig_image = rearrange(orig_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).to(img.device, dtype=img.dtype)
        img = t * img + (1.0 - t) * orig_image

    img_ids = img_ids.to(img.device, dtype=img.dtype)
    txt = txt.to(img.device, dtype=img.dtype)
    txt_ids = txt_ids.to(img.device, dtype=img.dtype)
    vec = vec.to(img.device, dtype=img.dtype)
    prev_pred = torch.zeros_like(img)

    if hasattr(model, "guidance_in"):
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    else:
        guidance_vec = None  # Ignored for models without guidance

    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), desc="Sampling", total=len(timesteps) - 1):
        h = t_prev - t_curr  # Time step size
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        # Compute k1 (initial slope)
        pred1 = model_forward(
            model,
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        if i >= timestep_to_start_cfg:
            neg_pred1 = model_forward(
                model,
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                neg_mode=True,
            )
            pred1 = neg_pred1 + true_gs * (pred1 - neg_pred1)

        # Estimate img at t_prev using Euler step
        img_temp = img + h * pred1

        # Compute k2 (slope at the end)
        t_prev_vec = torch.full((img.shape[0],), t_prev, dtype=img.dtype, device=img.device)
        pred2 = model_forward(
            model,
            img=img_temp,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_prev_vec,
            guidance=guidance_vec,
        )

        if i >= timestep_to_start_cfg:
            neg_pred2 = model_forward(
                model,
                img=img_temp,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_prev_vec,
                guidance=guidance_vec,
                neg_mode=True,
            )
            pred2 = neg_pred2 + true_gs * (pred2 - neg_pred2)

        # Update img using the average slope
        img = img + (h / 2) * (pred1 + pred2)

        if callback is not None:
            unpacked = unpack(img.float(), height, width)
            callback(step=i, x=img, x0=unpacked, total_steps=len(timesteps) - 1)
        i += 1

    return img

# Euler (initial)
def denoise_euler(
    model,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    image2image_strength=None,
    orig_image = None,
    callback = None,
    width = 512,
    height = 512,
):
    i = 0

      #init_latents = rearrange(init_latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if image2image_strength is not None and orig_image is not None:

        t_idx = int((1 - np.clip(image2image_strength, 0.0, 1.0)) * len(timesteps))
        t = timesteps[t_idx]
        timesteps = timesteps[t_idx:]
        orig_image = rearrange(orig_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).to(img.device, dtype = img.dtype)
        img = t * img + (1.0 - t) * orig_image
    img_ids=img_ids.to(img.device, dtype=img.dtype)
    txt=txt.to(img.device, dtype=img.dtype)
    txt_ids=txt_ids.to(img.device, dtype=img.dtype)
    vec=vec.to(img.device, dtype=img.dtype)
    if hasattr(model, "guidance_in"):
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    else:
        # this is ignored for schnell
        guidance_vec = None
    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), desc="Sampling", total = len(timesteps)-1):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model_forward(
            model,
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        if i >= timestep_to_start_cfg:
            neg_pred = model_forward(
                model,
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                neg_mode = True,
            )
            pred = neg_pred + true_gs * (pred - neg_pred)
        img = img + (t_prev - t_curr) * pred

        if callback is not None:
            unpacked = unpack(img.float(), height, width)
            callback(step=i, x=img, x0=unpacked, total_steps=len(timesteps) - 1)
        i += 1

    return img

def denoise_controlnet(
    model,
    controlnets_container: None|List[ControlNetContainer],
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    #controlnet_cond,
    #sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    #controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    image2image_strength=None,
    orig_image = None,
    callback = None,
    width = 512,
    height = 512,
    #controlnet_start_step=0,
    #controlnet_end_step=None
):
    i = 0

    if image2image_strength is not None and orig_image is not None:
        t_idx = int((1 - np.clip(image2image_strength, 0.0, 1.0)) * len(timesteps))
        t = timesteps[t_idx]
        timesteps = timesteps[t_idx:]
        orig_image = rearrange(orig_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2).to(img.device, dtype = img.dtype)
        img = t * img + (1.0 - t) * orig_image

    img_ids = img_ids.to(img.device, dtype=img.dtype)
    txt = txt.to(img.device, dtype=img.dtype)
    txt_ids = txt_ids.to(img.device, dtype=img.dtype)
    vec = vec.to(img.device, dtype=img.dtype)
    for container in controlnets_container:
        container.controlnet_cond = container.controlnet_cond.to(img.device, dtype=img.dtype)
        container.controlnet.to(img.device, dtype=img.dtype)
    #controlnet.to(img.device, dtype=img.dtype)
    #controlnet_cond = controlnet_cond.to(img.device, dtype=img.dtype)

    if hasattr(model, "guidance_in"):
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    else:
        guidance_vec = None

    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), desc="Sampling", total=len(timesteps)-1):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        guidance_vec = guidance_vec.to(img.device, dtype=img.dtype)
        controlnet_hidden_states = None
        for container in controlnets_container:
            if container.controlnet_start_step <= i <= container.controlnet_end_step:
                block_res_samples = container.controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=container.controlnet_cond,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
            if controlnet_hidden_states is None:                
                controlnet_hidden_states = [sample * container.controlnet_gs for sample in block_res_samples]
            else:
                if len(controlnet_hidden_states) == len(block_res_samples):
                    for j in range(len(controlnet_hidden_states)):
                        controlnet_hidden_states[j] += block_res_samples[j] * container.controlnet_gs
            

        pred = model_forward(
            model,
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=controlnet_hidden_states
        )
        neg_controlnet_hidden_states = None
        if i >= timestep_to_start_cfg:
            for container in controlnets_container:
                if container.controlnet_start_step <= i <= container.controlnet_end_step:
                    neg_block_res_samples = container.controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=container.controlnet_cond,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    y=neg_vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
                    if neg_controlnet_hidden_states is None:
                        neg_controlnet_hidden_states = [sample * container.controlnet_gs for sample in neg_block_res_samples]
                    else:
                        if len(neg_controlnet_hidden_states) == len(neg_block_res_samples):
                            for j in range(len(neg_controlnet_hidden_states)):
                                neg_controlnet_hidden_states[j] += neg_block_res_samples[j] * container.controlnet_gs
                

            neg_pred = model_forward(
                model,
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=neg_controlnet_hidden_states,
                neg_mode=True,
            )
            pred = neg_pred + true_gs * (pred - neg_pred)
        img = img + (t_prev - t_curr) * pred

        if callback is not None:
            unpacked = unpack(img.float(), height, width)
            callback(step=i, x=img, x0=unpacked, total_steps=len(timesteps) - 1)
        i += 1
    return img

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
