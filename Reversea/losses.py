import torch

def histogram_equalization_loss(image, num_bins=256):
    """
    Compute histogram equalization loss to encourage uniform histogram distribution
    Args:
        image: Input tensor of shape (B, C, H, W)
        num_bins: Number of histogram bins
    Returns:
        Scalar loss value
    """
    batch_size, channels, height, width = image.shape
    loss = 0
    
    # Process each channel separately
    for c in range(channels):
        # Get current channel
        channel = image[:, c:c+1, :, :]
        
        # Calculate histogram for this channel
        hist = torch.histc(channel, bins=num_bins, min=0, max=1)
        hist = hist / (height * width)  # Normalize histogram
        
        # Target uniform distribution
        target_hist = torch.ones_like(hist) / num_bins
        
        # Calculate KL divergence between actual and target histogram
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        hist = hist + eps
        target_hist = target_hist + eps
        
        # KL divergence loss
        loss += torch.sum(target_hist * torch.log(target_hist / hist))
        
    return loss / channels


def color_constancy_loss(image):
    """
    Computes the Color Constancy Loss for an input image.

    Args:
    - image: Tensor of shape (B, C, H, W) where B = batch size, 
             C = number of channels (3 for RGB), H = height, W = width.

    Returns:
    - loss: Color Constancy Loss (scalar).
    """
    # Ensure the image has 3 channels (RGB)
    if image.shape[1] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

    # Calculate mean for each color channel
    mu_R = torch.mean(image[:, 0, :, :], dim=(1, 2))  # Mean of Red channel
    mu_G = torch.mean(image[:, 1, :, :], dim=(1, 2))  # Mean of Green channel
    mu_B = torch.mean(image[:, 2, :, :], dim=(1, 2))  # Mean of Blue channel

    # Compute the loss components
    loss_RG = ((mu_R - mu_G) / (mu_R + mu_G + 1e-6))**2
    loss_GB = ((mu_G - mu_B) / (mu_G + mu_B + 1e-6))**2
    loss_BR = ((mu_B - mu_R) / (mu_B + mu_R + 1e-6))**2

    # Sum the losses and take the mean over the batch
    loss = torch.mean(loss_RG + loss_GB + loss_BR)
    
    return loss



