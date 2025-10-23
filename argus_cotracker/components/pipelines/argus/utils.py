from diffusers import UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder
from transformers import CLIPVisionModelWithProjection

from argus_cotracker.components.pipelines.argus.custom_argus_pipeline import StableVideoDiffusionPipelineCustom


def get_models_custom(args, accelerator, weight_dtype):
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_name_or_path=args.unet_path,
        subfolder="unet",
    )

    # feature_extractor = CLIPImageProcessor.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    # )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    pipeline = StableVideoDiffusionPipelineCustom.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(unet),
        image_encoder=accelerator.unwrap_model(image_encoder),
        vae=accelerator.unwrap_model(vae),
        revision=args.revision,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)

    return pipeline