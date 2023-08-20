"""
A handy utility for verifying SDXL image generation locally. 
To set up, first run a local cog server using:
   cog run -p 5000 python -m cog.server.http
Then, in a separate terminal, generate samples
   python samples.py
"""


import base64
import os
import sys

import requests


def gen(output_fn, **kwargs):
    if os.path.exists(output_fn):
        return

    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()

    try:
        datauri = data["output"][0]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
    except:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        sys.exit(1)

    with open(output_fn, "wb") as f:
        f.write(data)


def main():
    SCHEDULERS = [
        "DDIM",
        "DPMSolverMultistep",
        "HeunDiscrete",
        "KarrasDPM",
        "K_EULER_ANCESTRAL",
        "K_EULER",
        "PNDM",
    ]

    # gen(
    #     f"sample.txt2img.png",
    #     prompt="A studio portrait photo of a cat",
    #     num_inference_steps=25,
    #     guidance_scale=7,
    #     negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
    #     seed=1000,
    #     width=1024,
    #     height=1024,
    # )

    lora1 = "https://pbxt.replicate.delivery/5JPqMFc9u2qfbiDX4wjrAxhYyq7bIS5ehHzBWpi9vzY9JfviA/trained_model.tar"
    lora2 = "https://pbxt.replicate.delivery/EiSLCzkuLeR6HSQyzdDDR7PimKzIdQez59GuqAkzPY9ReqyiA/trained_model.tar"

    gen(
        f"sample.lora1.png",
        prompt="A studio portrait photo of a TOK",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=1024,
        height=1024,
        lora_url=lora1,
    )

    gen(
        f"sample.lora2.png",
        prompt="A studio portrait photo of a TOK",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=1024,
        height=1024,
        lora_url=lora2,
    )
    gen(
        f"sample.lora3.png",
        prompt="A studio portrait photo of a TOK",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=1024,
        height=1024,
        lora_url=lora1,
    )


if __name__ == "__main__":
    main()
