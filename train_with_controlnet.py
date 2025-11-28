# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import controlnet_params as params
from model import GradTTS_NS
from data import TextMelNoisyMelDataset, TextMelNoisyMelBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols

# Additional parameters
SNR_DB = params.snr_db
DEBUG = params.debug

train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale

if DEBUG:
    n_epochs = 1

if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    train_dataset = TextMelNoisyMelDataset(train_filelist_path, cmudict_path, add_blank,
                                   n_fft, n_feats, sample_rate, hop_length,
                                   win_length, f_min, f_max, SNR_DB)
    batch_collate = TextMelNoisyMelBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)
    test_dataset = TextMelNoisyMelDataset(valid_filelist_path, cmudict_path, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max, SNR_DB)
    test_loader = DataLoader(dataset=test_dataset, batch_size=params.test_size,
                             collate_fn=batch_collate, drop_last=True,
                             num_workers=4, shuffle=False)

    print('Initializing model...')
    model = GradTTS_NS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp,
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()
    base_state_dict = torch.load(params.base_weight_path)
    model.load_weights(params.base_weight_path)
    model.init_controlnet(base_state_dict)
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Logging test batch...')
    # test_batch = test_dataset.sample_test_batch(size=params.test_size)
    # for i, item in enumerate(test_batch):
    #     mel = item['y']
    #     logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
    #                      global_step=0, dataformats='HWC')
    #     save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    ckpt = model.state_dict()
    torch.save(ckpt, f=f"{log_dir}/grad_{0}.pt")
    print('Start training...')
    iteration = 0
    # containers for plotting
    train_dur_hist, train_prior_hist, train_diff_hist = [], [], []
    test_dur_hist, test_prior_hist, test_diff_hist = [], [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []


        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                c, c_lengths = batch['c'].cuda(), batch['c_lengths'].cuda()
                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths, c, c_lengths,
                                                                     out_size=out_size)
                loss = diff_loss # we only train the diffusion model, so only use the diffusion loss.
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                logger.add_scalar('training/duration_loss', dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)
                
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                
                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                    progress_bar.set_description(msg)
                
                iteration += 1


        avg_train_dur = np.mean(dur_losses)
        avg_train_prior = np.mean(prior_losses)
        avg_train_diff = np.mean(diff_losses)

        train_dur_hist.append(avg_train_dur)
        train_prior_hist.append(avg_train_prior)
        train_diff_hist.append(avg_train_diff)

        log_msg = ('Epoch %d | train: dur=%.3f prior=%.3f diff=%.3f ' %
                   (epoch, avg_train_dur, avg_train_prior, avg_train_diff))
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch % params.save_every > 0:
            continue

        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")

        model.eval()
        print('Evaluating on test set (compute_loss)...')
        test_dur_losses, test_prior_losses, test_diff_losses = [], [], []
        with torch.no_grad():
            with tqdm(test_loader, total=len(test_dataset) // batch_size) as progress_bar:
                for batch_idx, batch in enumerate(progress_bar):
                    x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                    y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                    c, c_lengths = batch['c'].cuda(), batch['c_lengths'].cuda()

                dur_l, prior_l, diff_l = model.compute_loss(x, x_lengths,
                                                             y, y_lengths,
                                                             c, c_lengths,
                                                             out_size=None)

                test_dur_losses.append(dur_l.item())
                test_prior_losses.append(prior_l.item())
                test_diff_losses.append(diff_l.item())

        avg_test_dur = np.mean(test_dur_losses)
        avg_test_prior = np.mean(test_prior_losses)
        avg_test_diff = np.mean(test_diff_losses)
        test_dur_hist.append(avg_test_dur)
        test_prior_hist.append(avg_test_prior)
        test_diff_hist.append(avg_test_diff)
        print(f'Test set: dur_loss={avg_test_dur:.3f}, prior_loss={avg_test_prior:.3f}, diff_loss={avg_test_diff:.3f}')

        logger.add_scalar('test/duration_loss', avg_test_dur, global_step=epoch)
        logger.add_scalar('test/prior_loss', avg_test_prior, global_step=epoch)
        logger.add_scalar('test/diffusion_loss', avg_test_diff, global_step=epoch)

    # -------- Plot after training ---------
    # epochs_range = range(1, len(train_dur_hist) + 1)
    # plt.style.use('default')
    # fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    # loss_names = ['Duration Loss', 'Prior Loss', 'Diffusion Loss']
    # for ax, train_hist, test_hist, name in zip(axes,
    #                                           [train_dur_hist, train_prior_hist, train_diff_hist],
    #                                           [test_dur_hist, test_prior_hist, test_diff_hist],
    #                                           loss_names):
    #     ax.plot(epochs_range, train_hist, label='Train')
    #     ax.plot(epochs_range, test_hist, label='Test')
    #     ax.set_ylabel(name)
    #     ax.grid(True, linestyle='--', alpha=0.4)
    #     ax.legend()
    # axes[-1].set_xlabel('Epoch')
    # plt.tight_layout()
    # plot_path = f"{log_dir}/train_test_loss.png"
    # plt.savefig(plot_path)
    # plt.close()
    # print(f'Saved loss plot to {plot_path}')

 # -------- Plot after training ---------
    plt.style.use('default')

    # 1) Train diffusion loss
    train_epochs = range(1, len(train_diff_hist) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(train_epochs, train_diff_hist, label='Train diffusion loss')
    plt.xlabel('Epoch')
    plt.ylabel('Diffusion Loss')
    plt.title('Train Diffusion Loss')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    train_plot_path = f"{log_dir}/train_diffusion_loss.png"
    plt.tight_layout()
    plt.savefig(train_plot_path)
    plt.close()
    print(f'Saved train diffusion loss plot to {train_plot_path}')

    # 2) Test diffusion loss (only if we have test data)
    if len(test_diff_hist) > 0:
        test_epochs = range(1, len(test_diff_hist) + 1)
        plt.figure(figsize=(8, 4))
        plt.plot(test_epochs, test_diff_hist, label='Test diffusion loss')
        plt.xlabel('Epoch')
        plt.ylabel('Diffusion Loss')
        plt.title('Test Diffusion Loss')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        test_plot_path = f"{log_dir}/test_diffusion_loss.png"
        plt.tight_layout()
        plt.savefig(test_plot_path)
        plt.close()
        print(f'Saved test diffusion loss plot to {test_plot_path}')