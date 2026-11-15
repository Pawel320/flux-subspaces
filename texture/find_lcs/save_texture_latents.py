import torch
import os
from diffusers import FluxPipeline

FLUX_PATH = "black-forest-labs/FLUX.1-dev"
OUT_DIR = "texture_latents"
IMAGE_SIZE = 512
NUM_SAMPLES_PER_TEXTURE = 5  # Liczba wariantów każdego materiału

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda"

# Ładujemy cały pipeline (potrzebujemy go do wygenerowania obrazka z promptu)
pipeline = FluxPipeline.from_pretrained(
    FLUX_PATH, torch_dtype=torch.bfloat16
).to(device)

vae = pipeline.vae

# Nasza lista tekstur (stanowiąca logiczną oś)
textures = [
    "mirror",
    "smooth glass",
    "polished metal",
    "shiny plastic",
    "matte paper",
    "rough wood",
    "porous stone",
    "bumpy concrete"
]

print(f"Rozpoczynam ekstrakcję dla {len(textures)} tekstur...")

for tex in textures:
    # Wymuszamy płaską powierzchnię wypełniającą cały kadr
    prompt = f"a flat {tex} wall surface, highly detailed macro photography, full frame"
    
    for i in range(NUM_SAMPLES_PER_TEXTURE):
        print(f"Generowanie i kodowanie: {tex} (Próbka {i+1}/{NUM_SAMPLES_PER_TEXTURE})")
        
        # 1. Wygenerowanie prawdziwego obrazka tekstury przez model
        image = pipeline(
            prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=20,
            guidance_scale=3.5
        ).images[0]
        
        # Zapisujemy podgląd w PNG, żebyś mógł sprawdzić, na co "patrzył" VAE
        image.save(os.path.join(OUT_DIR, f"{tex.replace(' ', '_')}_{i}.png"))
        
        # 2. Preprocessing z obrazka PIL do tensora (wymagane przez diffusers)
        image_tensor = pipeline.image_processor.preprocess(image).to(
            device=device, dtype=torch.torch.bfloat16
        )

        # 3. Właściwa kompresja VAE do Latent Space
        with torch.no_grad():
            dist = vae.encode(image_tensor)
            latents = dist.latent_dist.sample()

            # Normalizacja specyficzna dla architektury FLUX
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

            # Pakowanie tensorów do docelowego formatu
            latents = pipeline._pack_latents(
                latents,
                1,
                pipeline.transformer.config.in_channels // 4,
                2 * (IMAGE_SIZE // (pipeline.vae_scale_factor * 2)),
                2 * (IMAGE_SIZE // (pipeline.vae_scale_factor * 2)),
            )

        # 4. Zapis gotowego wektora
        out_path = os.path.join(OUT_DIR, f"{tex.replace(' ', '_')}_{i}.pt")
        torch.save(latents.cpu(), out_path)

print("Gotowe. Tensory oraz podglądy zapisano w folderze:", OUT_DIR)