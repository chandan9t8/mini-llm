import torch

def calculate_loss_batch(input_batch, target_batch, device, model):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())   #need to flatten both tensors to compute loss
    return loss

def calculate_loss_loader(data_loader, device, model, num_batches=None):
    total_loss = 0

    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calculate_loss_batch(input_batch, target_batch, device, model)
        total_loss += loss
    avg_loss = total_loss / num_batches
    return avg_loss

def train_model():
    pass



