from tqdm import tqdm
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def train(model, num_epochs, train_loader, test_loader, loss_function, forplot_img=None, optimizer = None, device = 'cuda'):
    print("training start")
    model.to(device)
    losses = []

    if optimizer is None:
        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2
        )


    for epoch in tqdm(range(num_epochs)):
        model.train()
        print(f"epoch: {epoch}")
        batch_losses = []
        for imgs, heatmaps, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)

            #print(f"imgs.shape: {imgs.shape}")            # imgs.shape: torch.Size([6, 640, 640, 3])
            #print(f"heatmaps.shape: {heatmaps.shape}")    # heatmaps.shape: torch.Size([6, 1, 160, 160])

            preds = model(imgs)

            loss = loss_function(preds, heatmaps)

            optimizer.zero_grad()
            loss.backward()\

            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(name, "→ grad is None")
            #     else:
            #         print(name, param.grad.mean().item())

            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        losses.append(epoch_loss)

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














