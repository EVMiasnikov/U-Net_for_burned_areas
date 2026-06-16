# Set seed for reproducibility
SEED = 42

# Import necessary libraries
import os
import shutil
from pathlib import Path
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import json

# Set environment variables before importing modules
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# Import necessary modules
import logging
import random
import numpy as np

# Set seeds for random number generators in NumPy and Python
np.random.seed(SEED)
random.seed(SEED)

# Import PyTorch
import torch
torch.manual_seed(SEED)
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import v2 as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import math
import matplotlib.gridspec as gridspec


from torchview import draw_graph

# Install torchmetrics for native metrics

import torchmetrics


# Import other libraries
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score




class NPYSegmentationDataset(Dataset):

    def __init__(self, root: str, split: str = "train", augmentation: bool = None):
        """
        Args:
            root:        path to OUT_DIR from patch_pipeline.py
            split:       "train" | "val" | "test"
            augmentation: True/False; defaults to True for train
        """
        self.img_dir     = Path(root) / split / "images"
        self.lbl_dir     = Path(root) / split / "labels"
        self.files       = sorted(self.img_dir.glob("*.npy"))
        self.augmentation = augmentation if augmentation is not None else (split == "train")

        if not self.files:
            raise RuntimeError(f"Patches not found: {self.img_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name  = self.files[idx].name

        # Read only one patch — not the entire dataset
        image = np.load(self.img_dir / name)              # (C, H, W) float32
        label = np.load(self.lbl_dir / name)              # (H, W)   uint8

        # HWC → CHW if patches are saved in old format
        if image.ndim == 3 and image.shape[-1] <= 20:
            image = image.transpose(2, 0, 1)

        # ── FIX: replace NaN/Inf with 0 before converting to tensor ──
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        # ─────────────────────────────────────────────────────────────

        image = torch.from_numpy(image.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        if self.augmentation:
            image, label = self._augment(image, label)

        return image, label

    def _augment(self, image: torch.Tensor, label: torch.Tensor):

        # Horizontal flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])

        # Vertical flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[0])

        # Rotate by 0 / 90 / 180 / 270°
        k = random.randint(0, 3)
        image = torch.rot90(image, k, dims=[1, 2])
        label = torch.rot90(label, k, dims=[0, 1])

        # Random channel dropout (1–2 channels)
        if random.random() < 0.3:
            n_drop = random.randint(1, 2)
            for ci in random.sample(range(image.shape[0]), n_drop):
                image[ci] = 0.0

        # Gaussian noise
        if random.random() < 0.5:
            image = torch.clamp(image + torch.randn_like(image) * 0.02, 0.0, 1.0)

        return image, label

def make_dataloaders(out_dir: str,
                     batch_size: int = 16,
                     num_workers: int = 2) -> dict:
    loaders = {}
    for split in ("train", "val", "test"):
        ds = NPYSegmentationDataset(out_dir, split)
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = (split == "train"),
        )
        print(f"{split:5s}: {len(ds):5d} patches → {len(loaders[split]):4d} batches/epoch")
    return loaders

def apply_colormap(label):

    label_vis = label.copy()

    # visualize ignore regions separately
    label_vis[label_vis == 255] = 2

    colormap = np.array([
        [0, 0, 0, 1],        # unburned (black)
        [1, 0, 0, 1],        # burned (red)
        [0, 0, 1, 1]   # ignore (blue)
    ])

    return colormap[label_vis.astype(int)]


def plot_sample_images(images, labels, num_samples=3):
    """
    Plot sample images with their corresponding segmentation masks.

    Args:
        images: Array of images (can be numpy array or list)
        labels: Array of labels (can be numpy array or list)
        num_samples: Number of samples to display
    """
    plt.figure(figsize=(15, 4*num_samples))

    # Select random sample indices
    num_available = len(images)
    indices = np.random.choice(num_available, min(num_samples, num_available), replace=False)

    for i, idx in enumerate(indices):
        # Get image and label
        if torch.is_tensor(images):
            # Convert from tensor (C, H, W) to numpy (H, W, C)
            image = images[idx].permute(1, 2, 0).cpu().numpy()
        else:
            image = images[idx]

        if torch.is_tensor(labels):
            label = labels[idx].cpu().numpy()
        else:
            label = labels[idx]

        # Display image
        plt.subplot(num_samples, 2, i*2 + 1)

        # plt.imshow(image)
        rgb = image[:, :, [9,8,7]]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        plt.imshow(rgb)

        plt.title(f'Image {i+1}', fontsize=14, fontweight='bold')
        plt.axis('off')

        # Display coloured segmentation label
        plt.subplot(num_samples, 2, i*2 + 2)
        colored_label = apply_colormap(label)
        plt.imshow(colored_label)
        plt.title(f'Label {i+1}', fontsize=14, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

class UNetBlock(nn.Module):
    """
    UNet convolutional block with multiple Conv-BN-ReLU stacks.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, num_convs=2):
        """
        Initialise a UNet block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            num_convs: Number of convolutional layers in the block
        """
        super().__init__()

        layers = []
        for i in range(num_convs):
            # Each iteration adds: Conv2D -> BatchNorm -> ReLU
            layers.extend([
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2  # Same padding to maintain spatial dimensions
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the block."""
        return self.block(x)


class UNet(nn.Module):
    """
    UNet architecture for semantic segmentation.

    Architecture:
        - Encoder: Downsampling path with pooling
        - Bottleneck: Deepest layer
        - Decoder: Upsampling path with skip connections
    """

    def __init__(self, in_channels=16, num_classes=2):
        """
        Initialise the UNet model.

        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes for segmentation
        """
        super().__init__()
        self.in_channels = in_channels # Store in_channels as an attribute

        # Encoder (downsampling path)
        self.down_block1 = UNetBlock(in_channels, 32, num_convs=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_block2 = UNetBlock(32, 64, num_convs=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (deepest layer)
        self.bottleneck = UNetBlock(64, 128, num_convs=2)

        # Decoder (upsampling path)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_block1 = UNetBlock(128 + 64, 64, num_convs=2)  # 128 + 64 due to concatenation

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_block2 = UNetBlock(64 + 32, 32, num_convs=2)  # 64 + 32 due to concatenation

        # Output layer: 1x1 convolution to produce class logits
        self.output = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through UNet.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, num_classes, height, width)
        """
        # Encoder with skip connections
        d1 = self.down_block1(x)
        p1 = self.pool1(d1)

        d2 = self.down_block2(p1)
        p2 = self.pool2(d2)

        # Bottleneck
        bottleneck = self.bottleneck(p2)

        # Decoder with skip connections
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, d2], dim=1)  # Concatenate skip connection
        u1 = self.up_block1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)  # Concatenate skip connection
        u2 = self.up_block2(u2)

        # Output layer
        output = self.output(u2)

        return output
    

import json
# Training function
def train_segmentation_model(model, train_loader, val_dataset, val_loader, meta,
                             epochs=100, learning_rate=1e-3,
                             patience=30, num_classes=2, device='cuda',
                             writer=None, experiment_name="unet",
                             visualize_every=5):
    """
    Train a segmentation model with native torchmetrics for evaluation.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_dataset: Dataset for validation data (used for visualisation)
        val_loader: DataLoader for validation data
        epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimiser
        patience: Early stopping patience (number of epochs)
        num_classes: Number of segmentation classes
        device: Device to use for training (cuda/cpu)
        writer: TensorBoard SummaryWriter for logging
        experiment_name: Name for saving model checkpoints
        visualize_every: Visualise predictions every N epochs
        metadata: Metadata file for the experiment

    Returns:
        Tuple of (trained_model, history_dict)
    """
    # Setup loss function and optimiser
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Setup torchmetrics for IoU calculation
    # JaccardIndex (aka IoU) with multiclass mode
    train_iou_metric = torchmetrics.JaccardIndex(
        task='multiclass',
        num_classes=num_classes,
        ignore_index=255,  # Ignore unlabeled regions
        average='weighted'  # Average across classes
    ).to(device)

    val_iou_metric = torchmetrics.JaccardIndex(
        task='multiclass',
        num_classes=num_classes,
        ignore_index=255,
        average='weighted'
    ).to(device)

    # Per-class IoU metrics for detailed analysis
    train_iou_per_class = torchmetrics.JaccardIndex(
        task='multiclass',
        num_classes=num_classes,
        ignore_index=255,
        average='none'  # Return IoU for each class
    ).to(device)

    val_iou_per_class = torchmetrics.JaccardIndex(
        task='multiclass',
        num_classes=num_classes,
        ignore_index=255,
        average='none'
    ).to(device)

    # Initialise training history
    history = {
        'loss': [], 'accuracy': [], 'mean_iou': [],
        'val_loss': [], 'val_accuracy': [], 'val_mean_iou': []
    }

    # Early stopping variables
    best_val_mean_iou = 0.0
    patience_counter = 0
    best_epoch = 0
    best_model_state = None

    # Get a fixed sample for visualisation during training
    # val_iter = iter(val_loader)
    # sample_batch_images, sample_batch_labels = next(val_iter)



    # for i in range(len(val_dataset)):

    #   _, lbl = val_dataset[i]

    #   if (lbl == 1).sum() > 500 and (lbl == 0).sum() > 3000:
    #       sample_idx = i
    #       break

    # sample_image, sample_label = val_dataset[sample_idx]

    # sample_image = sample_image.unsqueeze(0).to(device)
    # sample_label = sample_label.cpu().numpy()



    sample_idx = 0
    for i in range(min(20, len(val_dataset))):
        _, lbl = val_dataset[i]
        if (lbl == 1).sum() > 500 and (lbl == 0).sum() > 500:
            sample_idx = i
            break

    sample_image, sample_label = val_dataset[sample_idx]
    sample_label = sample_label.cpu().numpy()
    sample_image = sample_image.unsqueeze(0).to(device)
    # sample_image, sample_label = val_dataset[0]
    # sample_label = sample_label.cpu().numpy()
    # sample_image = sample_image.unsqueeze(0).to(device)






    # sample_image = sample_batch_images[0:1].to(device)
    # sample_label = sample_batch_labels[0].cpu().numpy()

    # sample_image_display = sample_batch_images[0].permute(1, 2, 0).cpu().numpy()
    # RGB visualization from multispectral channels
    sample_image_display = sample_image[0][[9,8,7], :, :]
    sample_image_display = sample_image_display.permute(1,2,0).cpu().numpy()

    # Normalize for plotting
    sample_image_display = (
        sample_image_display - sample_image_display.min()
    ) / (
        sample_image_display.max() - sample_image_display.min() + 1e-8
    )

    # Create colourmap for visualisation
    # colormap = create_segmentation_colormap(num_classes)

    print("Starting training with torchmetrics...")
    print("="*60)

    # print("Entering epoch loop...")
    for epoch in range(epochs):
        # print(f"Epoch {epoch+1} started")
        # ==================== TRAINING PHASE ====================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Reset metrics for this epoch
        train_iou_metric.reset()
        train_iou_per_class.reset()

        # for batch_images, batch_labels in train_loader:
        for i, (batch_images, batch_labels) in enumerate(train_loader):
            # print(f"  Batch {i} loaded")
            # Move data to device
            batch_images = batch_images.to(device)
            # print(f"  Batch {i} on GPU")
            batch_labels = batch_labels.to(device)

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                # print("Forward pass...")
                outputs = model(batch_images)
                # print(f"Output shape: {outputs.shape}")
                loss = criterion(outputs, batch_labels)
                # print(f"Loss: {loss.item()}")

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate metrics
            train_loss += loss.item() * batch_images.size(0)
            preds = outputs.argmax(dim=1)

            # train_correct += (preds == batch_labels).sum().item()
            # train_total += batch_labels.numel()
            valid_mask = batch_labels != 255
            train_correct += ((preds == batch_labels) & valid_mask).sum().item()
            train_total += valid_mask.sum().item()

            # Update IoU metric (torchmetrics handles everything)
            train_iou_metric.update(preds, batch_labels)
            train_iou_per_class.update(preds, batch_labels)

        # Compute epoch metrics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total
        train_mean_iou = train_iou_metric.compute().item()  # Get final IoU

        # ==================== VALIDATION PHASE ====================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Reset validation metrics
        val_iou_metric.reset()
        val_iou_per_class.reset()

        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(batch_images)
                    loss = criterion(outputs, batch_labels)

                # Accumulate metrics
                val_loss += loss.item() * batch_images.size(0)
                preds = outputs.argmax(dim=1)

                # val_correct += (preds == batch_labels).sum().item()
                # val_total += batch_labels.numel()
                valid_mask = batch_labels != 255
                val_correct += ((preds == batch_labels) & valid_mask).sum().item()
                val_total += valid_mask.sum().item()

                # Update IoU metric
                val_iou_metric.update(preds, batch_labels)
                val_iou_per_class.update(preds, batch_labels)

        # Compute validation metrics
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        val_mean_iou = val_iou_metric.compute().item()


        # ==================== RECORD HISTORY ====================
        history['loss'].append(epoch_train_loss)
        history['accuracy'].append(epoch_train_acc)
        history['mean_iou'].append(train_mean_iou)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)
        history['val_mean_iou'].append(val_mean_iou)

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
            writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
            writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
            writer.add_scalar('Accuracy/Val', epoch_val_acc, epoch)
            writer.add_scalar('MeanIoU/Train', train_mean_iou, epoch)
            writer.add_scalar('MeanIoU/Val', val_mean_iou, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            # Log per-class IoU to TensorBoard
            train_per_class = train_iou_per_class.compute()
            val_per_class = val_iou_per_class.compute()
            for c in range(num_classes):
                writer.add_scalar(f'IoU_PerClass_Train/Class_{c}', train_per_class[c].item(), epoch)
                writer.add_scalar(f'IoU_PerClass_Val/Class_{c}', val_per_class[c].item(), epoch)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Loss: {epoch_train_loss:.4f} | "
              f"Acc: {epoch_train_acc:.4f} | "
              f"IoU: {train_mean_iou:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.4f} | "
              f"Val IoU: {val_mean_iou:.4f}")

        # ==================== VISUALISE PREDICTIONS ====================
        if epoch % visualize_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                pred = model(sample_image)
                y_pred = pred.argmax(dim=1)[0].cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(16, 4))

            # Input image
            axes[0].imshow(sample_image_display)
            axes[0].set_title("Input Image", fontsize=13, fontweight='bold')
            axes[0].axis('off')

            # Ground truth
            colored_label = (apply_colormap(sample_label) * 255).astype(np.uint8)

            axes[1].imshow(colored_label)
            axes[1].set_title("Ground Truth", fontsize=13, fontweight='bold')
            axes[1].axis('off')

            # Prediction
            colored_pred = (apply_colormap(y_pred) * 255).astype(np.uint8)


            axes[2].imshow(colored_pred)
            axes[2].set_title(f"Prediction (Epoch {epoch+1})", fontsize=13, fontweight='bold')
            axes[2].axis('off')

            plt.tight_layout()
            if epoch % 30 == 0:
                plt.savefig(f"experiments/{experiment_name}/training_epoch_{epoch:03d}.png")
            plt.show()
            plt.close()

        if epoch % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_mean_iou,
                'history': history
            }

            torch.save(
                checkpoint,
                f"models/checkpoints/{experiment_name}/epoch_{epoch:03d}.pt"
            )
        # ==================== EARLY STOPPING ====================
        if val_mean_iou > best_val_mean_iou:
            best_val_mean_iou = val_mean_iou
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_mean_iou,
                'history': history
            }

            torch.save(
                checkpoint,
                f"models/final/{experiment_name}_best.pt"
            )

            # save metadata
            metadata = {
                "experiment_name": experiment_name,
                "best_epoch": best_epoch,
                "best_val_iou": float(best_val_mean_iou),
                "num_classes": num_classes,
                "in_channels": model.in_channels if hasattr(model, "in_channels") else None
            }
            
            for band_key in ('ps_bands', 's2_bands', 'ls_bands'):
                if band_key in meta:
                    metadata[band_key] = meta[band_key]

            with open(f"models/final/{experiment_name}.json", "w") as f:
                json.dump(metadata, f, indent=2)

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

    print("="*60)
    print("Training finished!")

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model from epoch {best_epoch+1} (Val Mean IoU: {best_val_mean_iou:.4f})")

    if writer is not None:
        writer.close()

    return model, history

# Visualising test predictions
def plot_triptychs_from_loader(test_loader, model, num_samples=5, device='cuda', experiment_name="default_experiment"):
    """
    Plot input image, ground truth, and prediction side by side using DataLoader.

    Args:
        test_loader: DataLoader containing test data
        model: Trained model for predictions
        num_samples: Number of samples to visualise
        device: Device to use for inference
        experiment_name: Name of the current experiment for saving plots.
    """
    model.eval()

    # colormap = create_segmentation_colormap(NUM_CLASSES)
    samples_shown = 0

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_size = batch_images.size(0)

            for i in range(batch_size):
                if samples_shown >= num_samples:
                    return

                # --- Data Prep ---
                # Extract single sample
                image = batch_images[i:i+1].to(device)
                label = batch_labels[i].cpu().numpy()

                # Generate prediction
                pred = model(image)
                pred = pred.argmax(dim=1)[0].cpu().numpy()


                # Mask ignore-regions from ground truth
                pred[label == 255] = 255  # ← add this line

                # Convert image for display
                # image_display = batch_images[i].permute(1, 2, 0).cpu().numpy()
                image_display = batch_images[i][[7, 6, 5], :, :].permute(1, 2, 0).cpu().numpy()
                image_display = (image_display - image_display.min()) / (image_display.max() - image_display.min() + 1e-8)


                legend_elements = [
                    Patch(facecolor='black', label='Unburned'),
                    Patch(facecolor='red', label='Burned'),
                    Patch(facecolor='blue', label='Ignore')
                ]

                # Create figure with three subplots
                fig, axes = plt.subplots(1, 3, figsize=(16, 4))

                # Original image
                axes[0].imshow(image_display)
                axes[0].set_title("Original Image", fontsize=13, fontweight='bold')
                axes[0].axis('off')

                # Ground truth mask
                colored_label = (apply_colormap(label) * 255).astype(np.uint8) # Convert to uint8
                axes[1].imshow(colored_label)
                axes[1].set_title("Ground Truth Mask", fontsize=13, fontweight='bold')
                axes[1].axis('off')

                # Predicted mask
                colored_pred = (apply_colormap(pred) * 255).astype(np.uint8) # Convert to uint8
                axes[2].imshow(colored_pred)
                axes[2].set_title("Predicted Mask", fontsize=13, fontweight='bold')
                axes[2].axis('off')

                plt.tight_layout()
                fig.legend(
                    handles=legend_elements,
                    loc='lower center',
                    ncol=3,
                    bbox_to_anchor=(0.5, -0.05)
                )
                plt.tight_layout(rect=[0, 0.1, 1, 1])

                plt.savefig(f"experiments/{experiment_name}/triptych_sample_{samples_shown}.png")
                plt.show()
                plt.close(fig)

                samples_shown += 1

def find_representative_indices(dataset, num_samples_to_find=5, min_burned_pixels=100, min_unburned_pixels=100):
    """
    Finds indices of samples in a dataset that contain both burned and unburned areas,
    to ensure varied samples for visualization.

    Args:
        dataset: The dataset to search (e.g., test_dataset).
        num_samples_to_find: The maximum number of such indices to find.
        min_burned_pixels: Minimum number of 'burned' pixels (label 1) required.
        min_unburned_pixels: Minimum number of 'unburned' pixels (label 0) required.

    Returns:
        A list of indices of representative samples.
    """
    representative_indices = []
    current_idx = 0
    while len(representative_indices) < num_samples_to_find and current_idx < len(dataset):
        # We only need the label to filter
        _, label = dataset[current_idx]
        label_np = label.cpu().numpy()

        burned_count = np.sum(label_np == 1)
        unburned_count = np.sum(label_np == 0)

        # Ensure there are both burned and unburned pixels
        if burned_count >= min_burned_pixels and unburned_count >= min_unburned_pixels:
            representative_indices.append(current_idx)
        current_idx += 1

    if len(representative_indices) < num_samples_to_find:
        print(f"Warning: Only found {len(representative_indices)} samples with significant burned and unburned areas. Will fill with random samples if needed.")
        # If not enough representative samples are found, fill with random ones
        remaining_needed = num_samples_to_find - len(representative_indices)
        all_indices = list(range(len(dataset)))
        # Remove already selected indices to avoid duplicates
        available_indices = list(set(all_indices) - set(representative_indices))
        if remaining_needed > 0 and len(available_indices) > 0:
            random_fill_indices = np.random.choice(available_indices, min(remaining_needed, len(available_indices)), replace=False).tolist()
            representative_indices.extend(random_fill_indices)

    # Shuffle the found indices to get a random subset if more than `num_samples_to_find` were initially found
    # Or to randomize the order of the chosen ones
    np.random.shuffle(representative_indices)

    return representative_indices[:num_samples_to_find]


def plot_triptychs_from_loader_with_selection(test_dataset, model, num_samples=5, device='cuda', experiment_name="default_experiment"):
    """
    Plot input image, ground truth, and prediction side by side using DataLoader.
    This version selects samples that have both burned and unburned areas.

    Args:
        test_dataset: The test dataset (not loader) for indexed access.
        model: Trained model for predictions
        num_samples: Number of samples to visualise
        device: Device to use for inference
        experiment_name: Name of the current experiment for saving plots.
    """
    model.eval()

    # Find indices of samples with both burned (1) and unburned (0) areas
    selected_indices = find_representative_indices(test_dataset, num_samples_to_find=num_samples, min_burned_pixels=100)

    samples_shown = 0
    with torch.no_grad():
        for data_idx in selected_indices:
            if samples_shown >= num_samples:
                break

            # Extract single sample from dataset using index
            image_tensor, label_tensor = test_dataset[data_idx]
            image = image_tensor.unsqueeze(0).to(device) # Add batch dimension for model
            label = label_tensor.cpu().numpy()

            # Generate prediction
            pred = model(image)
            pred = pred.argmax(dim=1)[0].cpu().numpy()

            # Mask ignore-regions from ground truth in prediction
            pred[label == 255] = 255

            # Convert image for display
            image_display = image_tensor[[7, 6, 5], :, :].permute(1, 2, 0).cpu().numpy()
            image_display = (image_display - image_display.min()) / (image_display.max() - image_display.min() + 1e-8)

            legend_elements = [
                Patch(facecolor='black', label='Unburned'),
                Patch(facecolor='red', label='Burned'),
                Patch(facecolor='blue', label='Ignore')
            ]

            # Create figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))

            # Original image
            axes[0].imshow(image_display)
            axes[0].set_title("Original Image", fontsize=13, fontweight='bold')
            axes[0].axis('off')

            # Ground truth mask
            # print(f"Sample {samples_shown} (Dataset Index {data_idx}): Ground Truth Mask unique values: {np.unique(label)}")
            colored_label = (apply_colormap(label) * 255).astype(np.uint8)
            axes[1].imshow(colored_label)
            axes[1].set_title("Ground Truth Mask", fontsize=13, fontweight='bold')
            axes[1].axis('off')

            # Predicted mask
            # print(f"Sample {samples_shown} (Dataset Index {data_idx}): Predicted Mask unique values: {np.unique(pred)}")
            colored_pred = (apply_colormap(pred) * 255).astype(np.uint8)
            axes[2].imshow(colored_pred)
            axes[2].set_title("Predicted Mask", fontsize=13, fontweight='bold')
            axes[2].axis('off')

            plt.tight_layout()
            fig.legend(
                handles=legend_elements,
                loc='lower center',
                ncol=3,
                bbox_to_anchor=(0.5, -0.05)
            )
            plt.tight_layout(rect=[0, 0.1, 1, 1])

            plt.savefig(f"experiments/{experiment_name}/selected_triptych_sample_{samples_shown}.png")
            plt.show()
            plt.close(fig)

            samples_shown += 1


def plot_layer_outputs_from_loader(test_loader, model, num_samples=3, n_cols_mask=4, device='cuda', experiment_name="default_experiment"):
    """
    Plots the input image on the left and a grid of last-layer activations on the right.
    Includes a colorbar for probability maps and saves the figure.
    """
    class_names = ["Unburned", "Burned"] # Assuming NUM_CLASSES = 2
    model.eval()
    samples_shown = 0

    n_classes = len(class_names)

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_size = batch_images.size(0)

            for i in range(batch_size):
                if samples_shown >= num_samples:
                    return

                # --- Data Prep ---
                img_tensor = batch_images[i:i+1].to(device)

                # Inference
                output = model(img_tensor)
                pred_probs = torch.softmax(output, dim=1)
                pred_numpy = pred_probs[0].cpu().numpy() # Shape: [C, H, W]

                # Input image prep (channels 9, 8, 7 for RGB visualization as per user's notebook)
                image_display = batch_images[i][[9, 8, 7], :, :].permute(1, 2, 0).cpu().numpy()
                image_display = (image_display - image_display.min()) / (image_display.max() - image_display.min() + 1e-8)

                # --- Plotting ---
                # Determine the actual number of columns for probability maps to avoid empty space
                # If n_classes is small, don't use the full n_cols_mask from arguments
                effective_n_cols_mask = min(n_cols_mask, n_classes)
                n_rows_prob_maps = math.ceil(n_classes / effective_n_cols_mask)

                # Calculate total figure width based on the number of actual columns for prob maps
                # Input image: 1 unit width
                # Probability maps: effective_n_cols_mask units width
                fig_width = 6 + effective_n_cols_mask * 6 # Base width 6 for input image, 6 for each prob map
                fig_height = 5 + n_rows_prob_maps * 5 # Base height 5 for title, 5 for each row of prob maps

                fig = plt.figure(figsize=(fig_width, fig_height))

                # Main Grid: Left (Input) vs Right (Masks and Colorbar)
                gs = gridspec.GridSpec(1, 2, width_ratios=[1, effective_n_cols_mask], figure=fig)

                ax_left = fig.add_subplot(gs[0])
                ax_left.imshow(image_display)
                ax_left.set_title("Input Image", fontsize=14, fontweight='bold')
                ax_left.axis('off')

                # Create a subgrid for the probability maps and the colorbar
                if n_classes > 0:
                    gs_right_main = gridspec.GridSpecFromSubplotSpec(n_rows_prob_maps + 1, effective_n_cols_mask, subplot_spec=gs[1],
                                                                      height_ratios=[1]*n_rows_prob_maps + [0.1], # Small height for colorbar row
                                                                      wspace=0.1, hspace=0.3)
                else: # Handle case with no classes to plot (unlikely, but robust)
                     gs_right_main = gridspec.GridSpecFromSubplotSpec(1, effective_n_cols_mask, subplot_spec=gs[1], wspace=0.1, hspace=0.3)


                im = None # To store the image object for colorbar reference
                for class_id in range(n_classes):
                    row = class_id // effective_n_cols_mask
                    col = class_id % effective_n_cols_mask

                    ax_mask = fig.add_subplot(gs_right_main[row, col])
                    # Display probability map with plasma colormap and define value range
                    im = ax_mask.imshow(pred_numpy[class_id], cmap='plasma', alpha=0.9, vmin=0, vmax=1) # vmin/vmax for probabilities
                    ax_mask.set_title(f"{class_names[class_id]} Probability", fontsize=10) # Clarify title
                    ax_mask.axis('off')

                # Add a single colorbar at the bottom of the probability maps grid
                if im is not None: # Only add colorbar if there were actual probability maps plotted
                    cbar_ax = fig.add_subplot(gs_right_main[n_rows_prob_maps, :]) # Takes the last row of the subgrid
                    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
                    cbar.set_label('Probability')

                plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make space for colorbar and overall title
                plt.savefig(f"experiments/{experiment_name}/layer_outputs_sample_{samples_shown}.png") # Save the figure
                plt.show()
                plt.close(fig) # Close the figure to free up memory
                print('\n')

                samples_shown += 1

# Contributors:
#  - Evgenii Miasnikov: evgenii.miasnikov@mail.polimi.it
#  - Ayman Mutasim Alfadul Abdelgadir: aymanmutasim@mail.polimi.it
#  - Eugenio Lomurno: eugenio.lomurno@polimi.it
#  - Alberto Archetti: alberto.archetti@polimi.it
#  - Roberto Basla: roberto.basla@polimi.it
#  - Carlo Sgaravatti: carlo.sgaravatti@polimi.it
# 
# Copyright and License:
# 
#    Original code:
#    Copyright 2025 Eugenio Lomurno, Alberto Archetti, Roberto Basla, Carlo Sgaravatti
# 
#    Modifications: 
#    Copyright 2026 Evgenii Miasnikov, Ayman Mutasim Alfadul Abdelgadir
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        https://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.