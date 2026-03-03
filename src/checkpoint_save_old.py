from tqdm import tqdm
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def train(model, num_epochs, save_every_n, train_loader, val_loader, loss_function, count_metrics, checkpoint_dir, forplot_img=None, optimizer = None, device = 'cuda'):
    print("training start")
    model.to(device)
    losses = []

    if optimizer is None:
        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2
        )


    #for epoch in tqdm(range(num_epochs)):
    for epoch in range(num_epochs):
        model.train()
        print(f"epoch: {epoch}", end=', ')
        batch_losses = []
        for imgs, heatmaps, pos_points, neg_points in tqdm(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)

            preds = model(imgs)
            loss = loss_function(preds, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

            #count_metrics(preds, heatmaps, pos_points, neg_points)

        epoch_loss = sum(batch_losses) / len(batch_losses)
        print(f"epoch_loss: {epoch_loss}")
        losses.append(epoch_loss)

        if epoch % save_every_n == 0:
            save_checkpoint(model, optimizer, epoch, epoch_loss, "heatmap_weighted_mse_loss", checkpoint_dir)

        if forplot_img is not None:
            forplot_img = forplot_img.to(device)
            forplot_predicted_heatmap = model(forplot_img)

            activ_forplot_predicted_heatmap = torch.sigmoid(forplot_predicted_heatmap)

            activ_forplot_predicted_heatmap_np = activ_forplot_predicted_heatmap.detach().cpu().numpy()
            plt.imshow(activ_forplot_predicted_heatmap_np[0, 0, :,:], cmap='autumn', interpolation='nearest')
            plt.colorbar()
            plt.title("Heatmap with color bar")
            plt.show()



    print("training done")
    return losses


def save_checkpoint(model, optimizer, epoch, loss, loss_f_name, checkpoint_dir, scheduler=None):
    """
    Docstring for save_checkpoint
    Saves a checkpoint of the model and its training parameters.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_name": optimizer.__class__.__name__,
        "epoch": epoch,
        "loss": loss,
        "loss_f_name": loss_f_name,
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    ch_path = checkpoint_dir / f"checkpoint_{epoch}.pt"
    torch.save(checkpoint, ch_path)

