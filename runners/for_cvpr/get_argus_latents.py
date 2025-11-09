from accelerate import Accelerator
from tqdm import tqdm

import torch

from conversions.mapper import Mappers
from runners.for_cvpr.settings import get_args, Settings, set_host_in_settings
from runners.model_utils import get_360_latents_model
from runners.pose_utils import VGGTPoseRunner
from runners.utils import get_dataset


def get_rotations(data, device, use_gt=False, offload_from_gpu=False):
    fov_x = data.fov_x[0]
    if use_gt:
        R_w2c = data.rotations[0]
    else:
        pose_runner = VGGTPoseRunner(device=device)
        # TODO must decide if im using these pred fov_x values anywhere
        pred_extrinsics, pred_fov_x = pose_runner.run(data.video[0].permute(0, 2, 3, 1))
        pred_rots = pose_runner.convert_to_tapvid360_format(pred_extrinsics[..., :3]).to(torch.float32)[0]
        R_w2c = pose_runner.align_to_ground_truth_start_rotations(pred_rots, data.rotations[0, 0])
        if offload_from_gpu:
            pose_runner.model.cpu()
            del pose_runner
            torch.cuda.empty_cache()

    return R_w2c.to(device), fov_x


def create_argus_inputs(persp_video_frames, rotations, mapper):
    conditional_video_equi, mask = mapper.image.perspective_image_to_equirectangular(persp_video_frames, rotations,
                                                                                     to_uint=False)
    conditional_video_equi = conditional_video_equi.clip(0, 1)
    # Put in the range of [-1, 1] for Argus
    conditional_video_equi = conditional_video_equi * 2 - 1
    conditional_video_equi = conditional_video_equi.permute(0, 3, 1, 2)
    mask = mask.unsqueeze(1)
    # Argus needs of shape 512, 1024
    conditional_video_equi = torch.nn.functional.interpolate(conditional_video_equi, size=(512, 1024),
                                                             mode='bilinear', align_corners=False)
    mask = torch.nn.functional.interpolate(mask.float(), size=(512, 1024), mode='nearest').bool()

    return conditional_video_equi, mask

def get_latents(condition_video_persp, conditional_video_equi, mask, accelerator, settings, latents_dir,
                 blend_frames=10, num_frames_batch=25):
    if not latents_dir.exists() or settings.force_get_latents:
        conditional_video_equi = conditional_video_equi.to(settings.weight_dtype)

        latents_model = get_360_latents_model(settings.paths.unet_path, accelerator, len(conditional_video_equi),
                                              accelerator.device, settings.weight_dtype, blend_frames=blend_frames,
                                              num_frames_batch=num_frames_batch)
        if len(condition_video_persp) > latents_model.argus_config.num_frames:
            print("WARNING: video longer than max frames for argus latents model, updating max frames")
            latents_model.argus_config.num_frames = len(condition_video_persp)
        argus_latents, _ = latents_model.forward_argus(condition_video_persp, conditional_video_equi, mask,
                                                       return_unet_latents=False)
        argus_latents = argus_latents[0]
        latents_dir.parent.mkdir(exist_ok=True, parents=True)
        torch.save({"latents": argus_latents.cpu()}, latents_dir)
    else:
        latents_data = torch.load(latents_dir)
        argus_latents = latents_data["latents"].to(accelerator.device)

    return argus_latents

@torch.no_grad()
def main(args, settings):
    dataset, dl = get_dataset(settings.paths.tapvid360_data_root, settings.ds_name, settings.specific_video_names)
    accelerator = Accelerator(mixed_precision='no')
    for data in tqdm(dl):
        data.video = data.video.to(accelerator.device, non_blocking=True)
        data.rotations = data.rotations.to(accelerator.device, non_blocking=True)
        normed_vid = (data.video.float() * 2) - 1
        R_w2c, fov_x = get_rotations(data, accelerator.device, use_gt=args.use_gt_rot)
        mapper = Mappers(data.video.shape[-1], data.video.shape[-2], settings.eq_height, fov_x=fov_x)
        input_eq_frames, input_masks = create_argus_inputs(data.video[0].permute(0, 2, 3, 1), R_w2c, mapper)
        out_dir = settings.paths.out_root / "argus_feats" / settings.ds_name / data.seq_name[
            0] / f"gt_poses-{args.use_gt_rot}"
        out_dir.mkdir(exist_ok=True, parents=True)
        get_latents(normed_vid[0].clone(), input_eq_frames, input_masks, accelerator, settings, out_dir / "latents.pth")
        torch.save({"rotations": R_w2c.cpu(), "fov_x": fov_x}, out_dir / "rotations.pth")


if __name__ == "__main__":
    args = get_args()
    settings = Settings()
    settings = set_host_in_settings(settings)
    main(args, settings)
