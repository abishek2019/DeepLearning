import torch
import time
import datetime
import random
import numpy as np
from datetime import datetime, timedelta
from Calculate_loss import cal_loss
from Test_and_Predict import test
from torchmetrics.audio import PerceptualEvaluationSpeechQuality as ps

def training_time_calc_new(training_mins):
    current_time = datetime.now()
    minutes_to_add = training_mins
    final_time = current_time + timedelta(minutes=minutes_to_add)
    return training_mins // 60, training_mins % 60, final_time.hour, final_time.minute

# Calculate batch PESQ
def calculate_pesq_for_batch_np(gen_outputs, sources):
    try:
        pesq_fn = ps(8000, 'nb', 12)
        score = pesq_fn(gen_outputs, sources)
    except:
        print(f'---------------------------------Except block reached PESQ2------------------------------------')
        score = 0.00001
    score = (score + 0.5) / 5
    return torch.full((len(gen_outputs),1), score)

# Gaussian noise
def gaussian_noise(mixture, i):
    std = 0.1
    min_amplitude, max_amplitude = 0.02, 0.05
    g_noise = torch.normal(mean=0, std=std, size=mixture.size())
    amplitude = np.random.uniform(min_amplitude, max_amplitude)
    mixture = mixture + (amplitude * g_noise)
    if i == 0:
        print(f'Gaussian Noise: {g_noise}')
        print(f'Mix shape: {mixture.shape} Amplitude: {amplitude}')
        print(f'Mix: {mixture}')
    return mixture

# Cutmix augmentation
def cutmix(mixture, sources, batch_size, frame_length, alpha=1.0):
    # Generate random indices for mixing
    rand_indices = torch.randperm(batch_size)
    # Generate binary mask
    # lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample().item()
    binary_mask = torch.zeros(batch_size, 1, frame_length)

    for i in range(batch_size):
        a = random.choice([0, 1])
        for j in range(frame_length):
            if a == 0:
                if j < (frame_length // 2):
                    binary_mask[i][0][j] = 1
            else:
                if j > (frame_length // 2):
                    binary_mask[i][0][j] = 1
    # print(f'binary mask: {binary_mask}')
    # Adjust mixture using the binary mask
    new_mixed_data = mixture * binary_mask + mixture[rand_indices] * (1 - binary_mask)
    # Adjust targets using the binary mask
    mixed_sources = sources * binary_mask + sources[rand_indices] * (1 - binary_mask)
    return new_mixed_data, mixed_sources

def train_new(args):
    mode = args['mode']
    device = args['device']
    data_loader = args['dataloader']
    dataloader_eval = args['dataloader_eval']
    model_gen = args['model_gen']
    optimizer_gen = args['optimizer_gen']
    scheduler_gen = args['scheduler_gen']
    model_discr = args['model_discr']
    optimizer_discr = args['optimizer_discr']
    loss_fn_discr = args['loss_fn_discr']
    adversarial_factor = args['adversarial_factor']

# -------------------------------------------------------Change here-------------------------------------------------------------------------
    a = 300
    epochs = 100
    best_si_snri = 0
    max_grad_norm = 5.0
    def model_save_location():
        dir_path, name = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Saved_Models/Libri-2Mix_Saved_models/8k_min_100_clean/NonGAN/B32_g16k_s4', '/Final_OrgNonGAN_part2_g16k_s4'
        return dir_path + name
# --------------------------------------------------------------------------------------------------------------------------------------------

    g_adv_score_means, d_real_score_means, d_fake_score_means, d_train_PESQ_score_means = [], [], [], []                # Epoch last-batch wise
    g_snr_losses, g_adv_losses, g_total_losses, d_real_losses, d_fake_losses, d_total_losses = [], [], [], [], [], []   # Epoch last-batch wise

    model_gen.train()
    if mode != 'Baseline':
        model_discr.train()

    print('TRAINING')
    count = 0
    for epoch in range(epochs):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'EPOCH: {epoch + 1}')
        print(f'Current Time: {current_time}')
        start = time.time()
        for i, batch in enumerate(data_loader):
            # 1. Get the data
            mixture, lens, sources = batch
            mixture = mixture.unsqueeze(1)
            
            # # Gaussian noise
            # mixture = gaussian_noise(mixture, i)
            # # Cutmix augmentation
            # mixture, sources = cutmix(mixture, sources, mixture.size(0), mixture.size(2), alpha=1.0)

            mixture = mixture.to(device)
            sources = sources.to(device)

            # 2. Zero gradient
            optimizer_gen.zero_grad()

            # GENERATOR
            # 3. Forward pass for generator/Conv-TasNet
            separated_sources = model_gen(mixture)

            # Separation loss (SI-SNR) for the ConvTasNet
            gen_loss, max_snr, separated_sources, reordered_separated_sources = cal_loss(separated_sources, sources, lens, device)

# -------------------------------------------------Non GAN------------------------------------------------------------------
            if mode == 'Baseline':

                # 4. Backward pass for generator/Conv-TasNet
                gen_total_loss = gen_loss
                gen_total_loss.backward()

                # Grad clipping
                torch.nn.utils.clip_grad_norm_(model_gen.parameters(), max_grad_norm)

                # 5. Optimize
                optimizer_gen.step()

#----------------------------------------------LSGAN ORIGINAL METHOD -------------------------------------------------------------
            elif mode == 'LSGAN':
                # For Vanilla GAN, change Loss fn to BCE and use sigmoid in the last layer of Discriminator (to ensure op in range 0-1/Prob that op belongs to 0/1 class) .

                optimizer_discr.zero_grad()

                # DISCRIMINATOR
                # Here for discriminator, real data is clean source and fake data is generated/separated source.

                # 3a. Forward pass for REAL Data

                real_input = sources
                real_score = model_discr(real_input)
                real_target_score = torch.ones_like(real_score)
                real_loss = loss_fn_discr(real_score, real_target_score)

                # 3b. Forward pass for FAKE/separated Data
                fake_input = reordered_separated_sources.detach()
                fake_score = model_discr(fake_input)
                fake_target_score = torch.zeros_like(fake_score)
                fake_loss = loss_fn_discr(fake_score, fake_target_score)

                # 4. Backward pass for Discriminator
                discr_total_loss = (real_loss + fake_loss) * 0.5
                # if count !=0 and count % 1 == 0:
                discr_total_loss.backward()

                # 5. Optimize Discr
                optimizer_discr.step()

                # GENERATOR

                # 3. Adversarial pass / Forward pass FAKE/separated data from Gen to discr
                adversarial_ip = reordered_separated_sources
                adversarial_score = model_discr(adversarial_ip)
                adversarial_target_score = torch.ones_like(adversarial_score)
                # Adversarial loss for Gen
                adversarial_loss = loss_fn_discr(adversarial_score, adversarial_target_score)  # Fake labels are real for Generator cost

                # 4. Backward pass for Gen (ConvTasNet)
                gen_total_loss = gen_loss + (adversarial_factor * adversarial_loss)
                gen_total_loss.backward()

                # Gradient clipping with maximum L2-norm of 5
                torch.nn.utils.clip_grad_norm_(model_gen.parameters(), max_norm=max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model_discr.parameters(), max_norm=2.0)

                # 5. Optimize Gen
                optimizer_gen.step()

# --------------------------------PESQGAN/MetricGAN-------------------------------------------------------------------------------
            elif mode == 'MetricGAN':

                optimizer_discr.zero_grad()

                # DISCRIMINATOR
                # Here for discriminator, real data is cat(clean,clean) source and fake data is cat(generated/separated, clean) source.

                # 3a. Forward pass for REAL Data
                real_input = torch.cat((sources, sources), dim=1)
                real_score = model_discr(real_input)
                real_target_score = torch.ones_like(real_score)
                real_loss = loss_fn_discr(real_score, real_target_score)

                # 3b. Forward pass for FAKE Data
                fake_input = torch.cat((sources, reordered_separated_sources.detach()), dim=1)
                fake_score = model_discr(fake_input)
                # *. PESQ score (is the target score for FAKE/generated data)
                train_PESQ_score = calculate_pesq_for_batch_np(reordered_separated_sources, sources).to(device)     # = fake_target_score
                fake_loss = loss_fn_discr(fake_score, train_PESQ_score)                                             # = Metric loss
                # 4. Backward pass for Discr
                discr_total_loss = (real_loss + fake_loss) * 0.5      #????
                # if count !=0 and count % 1 == 0:
                discr_total_loss.backward()

                # 5. Optimize Discr
                optimizer_discr.step()

                # GENERATOR

                # 3. Adversarial pass / Forward pass FAKE/separated data from Gen to discr
                adversarial_ip = torch.cat((sources, reordered_separated_sources), dim=1)
                adversarial_score = model_discr(adversarial_ip)
                adversarial_target_score = torch.ones_like(adversarial_score)
                # Adversarial loss for Gen
                adversarial_loss = loss_fn_discr(adversarial_score, adversarial_target_score) # Fake labels are real for Generator cost

                # 4. Backward pass for Gen (ConvTasNet)
                gen_total_loss = gen_loss + (adversarial_factor * adversarial_loss)
                gen_total_loss.backward()

                # Gradient clipping with maximum L2-norm of 5
                torch.nn.utils.clip_grad_norm_(model_gen.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model_discr.parameters(), max_norm=2.0)

                # 5. Optimize Gen
                optimizer_gen.step()

            count += 1

# """B -----------------------------------------------------------------------------------------------------------------------------"""
            # Print multiple performance attributes during an epoch

            if epoch % 10 == 0 and (i + 1) % 500 == 0:
                print(f'Mixture Shape: {mixture.shape}')
                print(f'Sources Shape: {sources.shape}')
                if mode != 'Baseline':
                    print(f'Adv ip shape (same as separated/or concatenated): {adversarial_ip.shape}')
                    print(f'Adv score shape (same as batch size): {adversarial_score.shape}\nop: {adversarial_score}')
                    print(f'Real score: {real_score}')

            if (i + 1) % a == 0:
                total = 0
                training_time = (((time.time() - start) / 60) * (1 / a) * ((len(data_loader) - (i + 1)) + (len(data_loader)) * (epochs - (epoch + 1))))
                total_hrs, total_mins, final_hr, final_min = training_time_calc_new(training_time)
                for r in range(max_snr.shape[0]):
                    total += max_snr[r].item()
                avg_snr = total / (max_snr.shape[0])
                print(f'Epoch: {epoch + 1}\tBatch: {i + 1}')
                print( f'\ttime/batch: {(time.time() - start) / 60:.3f}m per {a} batch\tTime left: {total_hrs} hours {total_mins:.1f} mins\tEst. finish time: {final_hr}: {final_min}')
                print(f'\tSNR: {avg_snr:.3f}\tGen LOSS: {gen_loss.item():.3f}')
                start = time.time()
                if mode != 'Baseline':
                    print(f'\tAdversarial Loss: {adversarial_loss.item():.3f}\tTotal Gen Loss: {gen_total_loss.item():.3f}')
                    print(f'\tDISCR LOSS: real_loss: {real_loss.item():.5f}\tFake Loss: {fake_loss.item():.5f}\tTotal Discr Loss: {discr_total_loss.item():.5f}')
                    print('\tThis Batch-mean wise:')
                    print(f'\tGen Adv score mean(~1): {adversarial_score.mean().item():.3f}\tDiscr Real score mean(~1): {real_score.mean().item():.3f}\tDiscr Fake score mean(~0): {fake_score.mean().item():.3f} ')
                    if mode == 'MetricGAN':
                        print(f'\tPESQ target score/Fake target score (batch_Mean): Train: {train_PESQ_score.mean().item():.3f}\tReal: {(train_PESQ_score.mean().item()) * 5 - 0.451:.3f}')

            g_snr_losses.append(round(gen_loss.item(), 4))
            g_total_losses.append(round(gen_total_loss.item(), 4))
            if mode != 'Baseline':
                g_adv_losses.append(round(adversarial_loss.item(), 4))
                d_real_losses.append(round(real_loss.item(), 4))
                d_fake_losses.append(round(fake_loss.item(), 4))
                d_total_losses.append(round(discr_total_loss.item(), 4))

            if (i + 1) == len(data_loader) and mode != 'Baseline':
                    g_adv_score_means.append(round(adversarial_score.mean().item(), 3))
                    d_real_score_means.append(round(real_score.mean().item(), 3))
                    d_fake_score_means.append(round(fake_score.mean().item(), 3))
                    if mode == 'MetricGAN':
                        d_train_PESQ_score_means.append(round(train_PESQ_score.mean().item(), 3))

# --------------------------------------------------------------------------------------------------------------------------------------------
        # End of one epoch

        # Evaluate (epochwise) the MODEL
        args_eval = {'dataloader': dataloader_eval, 'model_gen': model_gen, 'device': device, 'find_sdr': False}
        current_si_snri = test(args_eval, val_check = True)

        # Saving MODELS
        if current_si_snri > best_si_snri:
            best_si_snri = current_si_snri
            print(f'\tBEST PERFORMING EPOCH: {epoch + 1}\tSI-SNRI: {best_si_snri}')
            torch.save(model_gen.state_dict(), model_save_location() + '_e100_best.pth')
            print('\tSAVED best model.')
        if (epoch + 1) == 25 or (epoch + 1) == 50 or (epoch + 1) == 75:
            if (epoch + 1) == 25:
                torch.save(model_gen.state_dict(), model_save_location() + '_e25.pth')
            elif (epoch + 1) == 50:
                torch.save(model_gen.state_dict(), model_save_location() + '_e50.pth')
            elif (epoch + 1) == 75:
                torch.save(model_gen.state_dict(), model_save_location() + '_e75.pth')
            print(f'\tSAVED {epoch + 1} epoch model.')

        # Learning rate adjust
        scheduler_gen.step(current_si_snri)
        print(f'Current Learning Rate: {optimizer_gen.param_groups[0]["lr"]}')

# ----------------------------------------------------------------------------------------------------------------------------------------
    # End of Training

    print('Training Finished')
    torch.save(model_gen.state_dict(), model_save_location() + '_e100.pth')

    if mode == 'Baseline':
        print(f'\nGen_total_loss: {g_total_losses}')
    else:
        print('\tEpoch last-batch wise:')
        print(f'\nGen_Adv_score_means: {g_adv_score_means}')
        print(f'\nDiscr_real_score_means: {d_real_score_means}')
        print(f'\nDiscr_fake_score_means: {d_fake_score_means}')
        if mode == 'MetricGAN':
            print(f'\nDiscr_train_PESQ_score_means: {d_train_PESQ_score_means}')
        losses_list = [g_snr_losses, g_adv_losses, g_total_losses, d_real_losses, d_fake_losses, d_total_losses]
        losses_names = ['g_snr_losses', 'g_adv_losses', 'g_total_losses', 'd_real_losses', 'd_fake_losses', 'd_total_losses']
        for i, my_list in enumerate(losses_list):
            file_name = f'{losses_names[i]}.txt'
            with open('/home/abishek/speech_proj/ConvTasNet_PyTorch/Losses_txt/NewNonGAN/' + file_name, 'w') as file:
                for item in my_list:
                    file.write(f"{item}\n")




