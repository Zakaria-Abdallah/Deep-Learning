# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:09:42 2024

"""


#%%
#Imoports

import os
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import math
#%%
# Hyperparameters etc.
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 5
NUM_WORKERS = 4
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DIR = "Train/Train"
TEST_DIR ="Test/Test"
DEVICE
#%%
#Layer Normalisation
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# Modified Simple Gate
class SimpleGate(nn.Module):
    def __init__(self):
        super(SimpleGate, self).__init__()

    def forward(self, x, split_ratio=1):

        channels = x.size(1)  # Get the number of channels in the input tensor

        split_1 = int(channels *(1 - split_ratio))

        split_2 = channels - split_1

        # Split the input tensor into two parts along the channel dimension
        x1 = x[:, :split_1, :, :]
        x2 = x[:, split_1:, :, :]

        # If split_2 is odd, pad it with ones
        if split_2 % 2 != 0:
            ones_padding = torch.ones(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
            x2 = torch.cat((x2, ones_padding), dim=1)
            split_2 += 1


        # Split the second part into two halves
        x2_1 = x2[:, :split_2 // 2, :, :]
        x2_2 = x2[:, split_2 // 2:, :, :]

        # Apply element-wise multiplication to the two halves of the second part
        gated_output = x2_1 * x2_2

        # Concatenate the first part with the gated output along the channel dimension
        return torch.cat((x1, gated_output), dim=1)
# NAF Block
class NAFNetBlock(nn.Module):
    def __init__(self, in_channels,DW_Expand=2, FFN_Expand=2,drop_out_rate=0., split_ratio1 = 1, split_ratio2 = 1):
        super(NAFNetBlock, self).__init__()

        # Define expansion factors for channels
        self.dw_expand = DW_Expand
        self.ffn_expand = FFN_Expand

        # Split ratios for SimpleGate
        self.split_ratio1 = split_ratio1
        self.split_ratio2 = split_ratio2

        # Pooling size for SCA block
        self.sca_pool_size = 1

        # Calculate output channels after first expansion
        out_channels_dw = in_channels * self.dw_expand
        out_channels_ffn = in_channels * self.ffn_expand

        # Calculate SCA channels

        split_11 = int(out_channels_dw *(1 - self.split_ratio1))
        split_12 = out_channels_dw - split_11

        sca_channels_dw = split_11 + (split_12 + 1)//2

        split_21 = int(out_channels_ffn *(1 - self.split_ratio2))
        split_22 = out_channels_ffn - split_21

        sca_channels_ffn = split_21 + (split_22 + 1)//2

        # 1x1 convolution increases the channel dimension.
        self.conv1 = nn.Conv2d(in_channels, out_channels_dw, kernel_size=1, stride = 1,padding=0,groups = 1,bias=True)

        # 3x3 convolution applies spatial depthwise convolution convolution.
        self.conv2 = nn.Conv2d(out_channels_dw, out_channels_dw, kernel_size=3, stride=1, padding=1,groups = out_channels_dw,bias=True)

        # 1x1 convolution to reduce the channels
        self.conv3 = nn.Conv2d(sca_channels_dw , in_channels, kernel_size=1,stride = 1,padding=0,groups = 1, bias=True)

        # Layer normalization
        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)


        # SCA Block
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.sca_pool_size),
            nn.Conv2d(sca_channels_ffn , sca_channels_ffn, kernel_size=1,stride = 1,padding=0, groups = 1, bias=True)
        )

        # SimpleGate instance
        self.sg = SimpleGate()


        # Additional convolutions
        self.conv4 = nn.Conv2d(in_channels, out_channels_ffn, kernel_size=1, stride = 1,padding=0,groups = 1,bias=True)
        self.conv5 = nn.Conv2d(sca_channels_ffn, in_channels, kernel_size=1,stride = 1,padding=0,groups = 1, bias=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        identity = x # Save input tensor for residual connection

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x, self.split_ratio1)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = identity + x * self.beta # First residual connection

        x = self.norm2(y)

        x = self.conv4(x)
        x = self.sg(x, self.split_ratio2)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma # Second residual connection
    
# NAFNet Architecture    
class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, bottleneck_num=1, enc_blk_nums=[], dec_blk_nums=[], split_ratio1 = 1, split_ratio2 = 1):
        super().__init__()

        # Initial convolutional layer
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        # Final convolutional layer
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        # Lists to store encoder, decoder, upsampling, and downsampling blocks
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            # Adding encoder blocks
            self.encoders.append(
                nn.Sequential(
                    *[NAFNetBlock(chan,split_ratio1 = split_ratio1,split_ratio2=split_ratio2) for _ in range(num)]
                )
            )
            # Adding downsampling layers
            self.downs.append(
                nn.Conv2d(chan, 2*chan, kernel_size = 2, stride = 2)
            )
            chan = chan * 2

        # Bottleneck blocks
        self.bottleneck = nn.Sequential(
            *[NAFNetBlock(chan,split_ratio1 = split_ratio1,split_ratio2=split_ratio2) for _ in range(bottleneck_num)]
        )

        for num in dec_blk_nums:
            # Adding upsampling layers
            self.ups.append(
                nn.Sequential(
                    #nn.ConvTranspose2d(chan, chan // 2, kernel_size=2, stride=2)
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2

            # Adding decoder blocks
            self.decoders.append(
                nn.Sequential(
                    *[NAFNetBlock(chan,split_ratio1 = split_ratio1,split_ratio2=split_ratio2) for _ in range(num)]
                )
            )

            self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
            # Check if image dimensions are divisible by a factor
            _, _, H, W = x.size()  # Extract the batch size, channels, height, and width
            pad_h = (self.padder_size - H % self.padder_size) % self.padder_size  # Calculate padding needed for height
            pad_w = (self.padder_size - W % self.padder_size) % self.padder_size  # Calculate padding needed for width
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))  # Apply padding to the input tensor
            return x

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        skip_connections = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skip_connections.append(x)
            x = down(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for decoder, up, skip_connection in zip(self.decoders, self.ups, skip_connections):
            x = up(x)
            x = x + skip_connection
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

#test
model = NAFNet(img_channel=3, width=16, bottleneck_num=1, enc_blk_nums=[2, 2, 2, 2], dec_blk_nums=[2, 2, 2, 2])
input_tensor = torch.randn(1, 3, 256, 256)
output = model(input_tensor)
print(output.shape)
#%%
# loading data

class NoisyDataset(Dataset):
    def __init__(self, directory, transform= None, noise_level=1):

        self.directory = directory
        self.image_filenames = [f for f in os.listdir(directory) if f.endswith('.jpg')]
        self.noise_level = noise_level
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_filenames[idx])
        
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
    
        # Générer du bruit avec la même forme que l'image, multiplié par le niveau de bruit
        noise = np.random.randn(*image.shape).astype(np.float32) * self.noise_level
    
        # Ajouter le bruit à l'image et s'assurer que les valeurs restent dans l'intervalle [0, 255]
        noisy_image = np.clip(image + noise, 0, 255)
    
        # Normaliser les valeurs entre 0 et 1 pour les deux images
        noisy_image /= 255.0
        image /= 255.0

        if self.transform is not None:
            image = self.transform(image=image)['image']
            noisy_image = self.transform(image=noisy_image)['image']

        return noisy_image, image

def create_data_loaders(directory_train, directory_test, transform=None, batch_size=32, noise_level=35, num_workers=4, pin_memory=True,):

    train_dataset = NoisyDataset(directory_train,transform, noise_level=noise_level)
    test_dataset = NoisyDataset(directory_test,transform, noise_level=noise_level)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

    return train_loader, test_loader

def update_learning_rate(schedulers, optimizer, current_iter, warmup_iter=-1):
    if current_iter > 1:
        for scheduler in schedulers:
            scheduler.step()
    # set up warm-up learning rate
    if current_iter < warmup_iter:
        # get initial lr for each group
        init_lr_g_l = [group['initial_lr'] for group in optimizer.param_groups]
        # modify warming-up learning rates
        # currently only support linearly warm up
        warm_up_lr_l = []
        for init_lr_g in init_lr_g_l:
            warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
        # set learning rate
        for param_group, lr in zip(optimizer.param_groups, warm_up_lr_l):
            param_group['lr'] = lr
            
transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2(),
        ],
    )
train_loader, test_loader = create_data_loaders(TRAIN_DIR, TEST_DIR, transform,BATCH_SIZE, 28, NUM_WORKERS, PIN_MEMORY)
#%%
# train

def train_fn(loader, model, optimizer, loss_fn, scaler, scheduler=None,warmup_iter=-1):

    model.train()  # Set the model to training mode
    total_loss = 0
    total_samples = 0

    loop = tqdm(loader, leave=True)  # Create a progress bar

    for batch_idx, (noisy_images, clean_images) in enumerate(loop):
        # Move data to the appropriate device
        noisy_images = noisy_images.to(DEVICE)
        clean_images = clean_images.float().to(DEVICE)

        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast():
            predictions = model(noisy_images)
            loss = loss_fn(predictions, clean_images)

        # Backward pass and optimization
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        update_learning_rate([scheduler], batch_idx, warmup_iter)


        # Accumulate loss
        batch_loss = loss.item() * noisy_images.size(0)
        total_loss += batch_loss
        total_samples += noisy_images.size(0)

        # Update the progress bar
        loop.set_postfix(loss=(total_loss / total_samples))

    return total_loss / total_samples
#%%
# Utils
def load_checkpoint(checkpoint_path, model, optimizer=None):
    print("=> Loading checkpoint")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint["state_dict"])
    
    # Load optimizer state if provided
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    start_epoch = checkpoint.get("epoch", 0)  # Get the epoch number, default to 0 if not present
    print(f"=> Loaded checkpoint (Epoch {start_epoch})")
    
    return start_epoch

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
#%%
# Plotting images to test noise

def plot_image(image):
  image_np = image[0].numpy()*255
  image_np=  np.transpose(image_np, (1, 2, 0))
  image_np = np.clip(image_np, 0, 255).astype(np.uint8)
  plt.imshow(image_np)
  plt.show()

i = 0
for (image , label) in train_loader:
  print(image.shape)
  plot_image(image)
  plot_image(label)
  i +=1
  if (i==1):
    break
#%%
# Validation 

def validate_fn(loader, model, loss_fn):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for noisy_images, clean_images in loader:
            noisy_images = noisy_images.to(DEVICE)
            clean_images = clean_images.float().to(DEVICE)
            predictions = model(noisy_images)

            loss = loss_fn(predictions, clean_images)
            total_loss += loss.item() * noisy_images.size(0)
            total_samples += noisy_images.size(0)

    avg_loss = total_loss / total_samples
    avg_psnr = -avg_loss

    return avg_loss, avg_psnr
#%%

# Loss function 

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, toY=False):
        super(PSNRLoss, self).__init__()
        self.loss_weight = loss_weight
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred/255.0, target/255.0
            #pass

        return self.loss_weight * 10 * torch.log10(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
    
class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.

    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7

    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1, ),
                 eta_min=0,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]

def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
torch.cuda.empty_cache()
#%%

train_losses = []
val_losses = []
# define model
model_NAF = NAFNet(width=32, enc_blk_nums = [2, 2, 4, 8],bottleneck_num=12,dec_blk_nums = [2, 2, 2, 2]).to(DEVICE)
optimizer_NAF = optim.Adam(model_NAF.parameters(),lr =LEARNING_RATE,betas=(0.9,0.9))

start_epoch = 0

if LOAD_MODEL:
    start_epoch = load_checkpoint("my_checkpoint3.pth.tar", model_NAF, optimizer_NAF)

scheduler = CosineAnnealingRestartLR(optimizer_NAF,[25, 25, 25, 25], restart_weights = [1, 0.8, 0.6, 0.4], eta_min=1e-7, last_epoch=start_epoch-1)
loss_NAF = PSNRLoss(toY=False)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(start_epoch, NUM_EPOCHS + start_epoch):
    train_loss = train_fn(train_loader, model_NAF, optimizer_NAF, loss_NAF, scaler)
    
    train_losses.append(train_loss)
    
    # save model
    checkpoint = {
        "state_dict": model_NAF.state_dict(),
        "optimizer":optimizer_NAF.state_dict(),
        "epoch": epoch + 1
        }
    save_checkpoint(checkpoint, filename="my_checkpoint3.pth.tar")
    
    val_loss, val_psnr = validate_fn(test_loader, model_NAF, loss_NAF)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss}, Val Loss: {val_loss},Val PSNR: {val_psnr}")
    
    val_losses.append(val_loss) 
#%%
plt.figure(figsize=(10, 5))
plt.plot(range(start_epoch + 1, start_epoch + 1 + len(train_losses)), -np.array(train_losses), label='Training PSNR', color='blue')
plt.plot(range(start_epoch + 1, start_epoch + 1 + len(val_losses)), -np.array(val_losses), label='Validation PSNR', color='orange')

# Add titles and labels
plt.title('Training and Validation PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()

# Show the plot
plt.show()
#%%
# Predict function
def predict(model, image):
    
    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Predict
        output = model(image)
    
    # Convert the output to numpy array
    output_image = output.squeeze(0).cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))  # HWC format

    # Denormalize the output image from [0, 1] to [0, 255]
    output_image = (output_image * 255).astype(np.uint8)

    return output_image

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


test_dataset = NoisyDataset(TEST_DIR,transform, noise_level=28)

random_idx = np.random.randint(0, len(test_dataset) - 1)
noisy_image, ground_truth = test_dataset[random_idx]

# Add batch dimension and move to device
noisy_image = noisy_image.unsqueeze(0).to(DEVICE)
ground_truth = ground_truth.unsqueeze(0).to(DEVICE)

# Predict the output
predicted_image = predict(model_NAF, noisy_image)

# Convert ground truth to numpy array
ground_truth = ground_truth.squeeze(0).cpu().numpy()
ground_truth = np.transpose(ground_truth, (1, 2, 0))  # HWC format
ground_truth = (ground_truth * 255).astype(np.uint8)

# Calculate PSNR
psnr_value = calculate_psnr(ground_truth, predicted_image)
print(f'PSNR: {psnr_value}')

# Display images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Noisy Image')
plt.imshow(noisy_image.squeeze(0).cpu().permute(1, 2, 0).numpy())

plt.subplot(1, 3, 2)
plt.title('Ground Truth')
plt.imshow(ground_truth)

plt.subplot(1, 3, 3)
plt.title('Predicted Image')
plt.imshow(predicted_image)
plt.show()

#%%
train_losses_Y = []
val_losses_Y = []
model_NAF = NAFNet(width=32, enc_blk_nums = [2, 2, 4, 8],bottleneck_num=12,dec_blk_nums = [2, 2, 2, 2]).to(DEVICE)
optimizer_NAF = optim.Adam(model_NAF.parameters(),lr =LEARNING_RATE,betas=(0.9,0.9))


start_epoch = 0

#if LOAD_MODEL:
#    start_epoch = load_checkpoint("my_checkpoint3.pth.tar", model_NAF, optimizer_NAF)

scheduler = CosineAnnealingRestartLR(optimizer_NAF,[25, 25, 25, 25], restart_weights = [1, 0.8, 0.6, 0.4], eta_min=1e-7, last_epoch=start_epoch-1)
loss_NAF = PSNRLoss(toY=True)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(start_epoch, NUM_EPOCHS + start_epoch):
    train_loss = train_fn(train_loader, model_NAF, optimizer_NAF, loss_NAF, scaler)

    train_losses_Y.append(train_loss)

    # save model
    #checkpoint = {
    #    "state_dict": model_NAF.state_dict(),
    #    "optimizer":optimizer_NAF.state_dict(),
    #    "epoch": epoch + 1
    #    }
    #save_checkpoint(checkpoint, filename="my_checkpoint3.pth.tar")

    val_loss, val_psnr = validate_fn(test_loader, model_NAF, loss_NAF)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss}, Val Loss: {val_loss},Val PSNR: {val_psnr}")

    val_losses_Y.append(val_loss)

#%%
plt.figure(figsize=(10, 5))
plt.plot(range(start_epoch + 1, start_epoch + 1 + len(train_losses_Y)), -np.array(train_losses_Y), label='Training PSNR', color='blue')
plt.plot(range(start_epoch + 1, start_epoch + 1 + len(val_losses_Y)), -np.array(val_losses_Y), label='Validation PSNR', color='orange')

# Add titles and labels
plt.title('Training and Validation PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()

# Show the plot
plt.show()

#%%

predicted_image_Y = predict(model_NAF, noisy_image)

psnr_value = calculate_psnr(ground_truth, predicted_image_Y)
print(f'PSNR: {psnr_value}')

# Display images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Ground Truth')
plt.imshow(ground_truth)

plt.subplot(1, 3, 2)
plt.title('Predicted Image')
plt.imshow(predicted_image)

plt.subplot(1, 3, 3)
plt.title('Predicted Image Y')
plt.imshow(predicted_image)

#%%
train_losses_05 = []
val_losses_05 = []
model_NAF_05 = NAFNet(width=32, enc_blk_nums = [2, 2, 4, 8],bottleneck_num=12,dec_blk_nums = [2, 2, 2, 2],split_ratio1=0.5,split_ratio2 = 0.5).to(DEVICE)
optimizer_NAF = optim.Adam(model_NAF_05.parameters(),lr =LEARNING_RATE,betas=(0.9,0.9))


start_epoch = 0

#if LOAD_MODEL:
#    start_epoch = load_checkpoint("my_checkpoint3.pth.tar", model_NAF_05, optimizer_NAF)

scheduler = CosineAnnealingRestartLR(optimizer_NAF,[25, 25, 25, 25], restart_weights = [1, 0.8, 0.6, 0.4], eta_min=1e-7, last_epoch=start_epoch-1)
loss_NAF = PSNRLoss(toY=True)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(start_epoch, NUM_EPOCHS + start_epoch):
    train_loss = train_fn(train_loader, model_NAF_05, optimizer_NAF, loss_NAF, scaler)

    train_losses_05.append(train_loss)

    # save model
    #checkpoint = {
    #    "state_dict": model_NAF_05.state_dict(),
    #    "optimizer":optimizer_NAF.state_dict(),
    #    "epoch": epoch + 1
    #    }
    #save_checkpoint(checkpoint, filename="my_checkpoint3.pth.tar")

    val_loss, val_psnr = validate_fn(test_loader, model_NAF_05, loss_NAF)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss}, Val Loss: {val_loss},Val PSNR: {val_psnr}")

    val_losses_05.append(val_loss)
#%%
plt.figure(figsize=(10, 5))
plt.plot(range(start_epoch + 1, start_epoch + 1 + len(train_losses_05)), -np.array(train_losses_05), label='Training PSNR', color='blue')
plt.plot(range(start_epoch + 1, start_epoch + 1 + len(val_losses_05)), -np.array(val_losses_05), label='Validation PSNR', color='orange')

# Add titles and labels
plt.title('Training and Validation PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()

# Show the plot
plt.show()

#%%
train_losses_025 = []
val_losses_025 = []
model_NAF_025 = NAFNet(width=32, enc_blk_nums = [2, 2, 4, 8],bottleneck_num=12,dec_blk_nums = [2, 2, 2, 2],split_ratio1=0.25,split_ratio2 = 0.25).to(DEVICE)
optimizer_NAF = optim.Adam(model_NAF_025.parameters(),lr =LEARNING_RATE,betas=(0.9,0.9))


start_epoch = 0

#if LOAD_MODEL:
#    start_epoch = load_checkpoint("my_checkpoint3.pth.tar", model_NAF_025, optimizer_NAF)

scheduler = CosineAnnealingRestartLR(optimizer_NAF,[25, 25, 25, 25], restart_weights = [1, 0.8, 0.6, 0.4], eta_min=1e-7, last_epoch=start_epoch-1)
loss_NAF = PSNRLoss(toY=True)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(start_epoch, NUM_EPOCHS + start_epoch):
    train_loss = train_fn(train_loader, model_NAF_025, optimizer_NAF, loss_NAF, scaler)

    train_losses_025.append(train_loss)

    # save model
    #checkpoint = {
    #    "state_dict": model_NAF_025.state_dict(),
    #    "optimizer":optimizer_NAF.state_dict(),
    #    "epoch": epoch + 1
    #    }
    #save_checkpoint(checkpoint, filename="my_checkpoint3.pth.tar")

    val_loss, val_psnr = validate_fn(test_loader, model_NAF_025, loss_NAF)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss}, Val Loss: {val_loss},Val PSNR: {val_psnr}")

    val_losses_025.append(val_loss)
#%%
plt.figure(figsize=(10, 5))
plt.plot(range(start_epoch + 1, start_epoch + 1 + len(train_losses_025)), -np.array(train_losses_025), label='Training PSNR', color='blue')
plt.plot(range(start_epoch + 1, start_epoch + 1 + len(val_losses_025)), -np.array(val_losses_025), label='Validation PSNR', color='orange')

# Add titles and labels
plt.title('Training and Validation PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()

# Show the plot
plt.show()
