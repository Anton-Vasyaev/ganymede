import torch


def calculate_dataset_accuracy(
    model, 
    batch_sampler,
    batch_processor,
    accuracy_func,
    postprocessing_func
):
    model.train(False)

    device = next(model.parameters()).device

    total_loss     = 0.0
    total_examples = 0

    with torch.no_grad():
        for idx in range(len(batch_sampler)):
            example = batch_sampler[idx]
            img_b, target_b = batch_processor(example)

            img_b, target_b = img_b.to(device), target_b.to(device)

            output = model(img_b)
            if not postprocessing_func is None: output = postprocessing_func(output)
            batch_size = img_b.shape[0]

            total_loss += accuracy_func(output, target_b) * batch_size

            total_examples += batch_size

            torch.cuda.empty_cache()

        error = total_loss / total_examples

    model.train(True)

    return error