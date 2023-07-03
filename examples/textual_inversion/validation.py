from loguru import logger
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import wandb
import numpy as np

def log_validation(text_encoder, tokenizer, unet, vae, accelerator, weight_dtype, epoch, model_checkpoint_id: str, model_checkpoint_revision: str, validation_prompt: str, num_images: int = 4, seed=0):
    logger.info(
        f"Running validation... \n Generating {num_images} images with prompt:"
        f" {validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        model_checkpoint_id,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=model_checkpoint_revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if seed is None else torch.Generator(device=accelerator.device).manual_seed(seed)
    images = []
    for _ in range(num_images):
        with torch.autocast("cuda"):
            image = pipeline(validation_prompt, num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()
    return images
