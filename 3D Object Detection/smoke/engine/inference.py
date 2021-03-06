import logging
from tqdm import tqdm

import torch

from smoke.utils import comm
from smoke.utils.timer import Timer, get_time_str
from smoke.data.datasets.evaluation import evaluate


def compute_on_dataset(model, data_loader, device, timer=None, add_depth=False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, depth = batch["images"], batch["targets"], batch["img_ids"], batch["depth"]
        images = images.to(device)
        if add_depth:
            depth = depth.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            output = model(images, targets, depth)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = output.to(cpu_device)
        results_dict.update(
            {img_id: output for img_id in image_ids}
        )
    return results_dict


def inference(
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,
        add_depth=False

):
    device = torch.device(device)
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer, add_depth)
    comm.synchronize()

    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    if not comm.is_main_process():
        return

    return evaluate(eval_type=eval_types,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder, )
