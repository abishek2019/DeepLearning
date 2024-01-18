import torch
import time
import datetime
from datetime import datetime, timedelta
from Calculate_loss import cal_loss
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

def training_time_calc(training_mins):
    current_time = datetime.now()
    minutes_to_add = training_mins
    final_time = current_time + timedelta(minutes=minutes_to_add)
    return training_mins // 60, training_mins % 60, final_time.hour, final_time.minute

def pad_zeroes(mixture, sources, lens, batch_size):
    frame_length = int(mixture.size(2))
    num_samples_to_pad = batch_size - mixture.size(0)
    zero_padding_mixture = torch.zeros(num_samples_to_pad, 1, frame_length)
    zero_padding_sources = torch.zeros(num_samples_to_pad, 2, frame_length)
    padding_lens = torch.full((num_samples_to_pad,), frame_length)
    mixture = torch.cat((mixture, zero_padding_mixture), dim=0)
    sources = torch.cat((sources, zero_padding_sources), dim=0)
    lens = torch.cat((lens, padding_lens), dim=0)
    return mixture, sources, lens


def get_factor(epoch):
    factor = 10
    # if epoch > 3 and epoch <= 7: factor = 0.1
    # elif epoch > 7 and epoch <= 30: factor = 0.01
    # elif epoch > 30 and epoch <= 60: factor = 0.001
    # elif epoch > 60: factor = 0.001
    return factor

import torch
from torch_pesq import PesqLoss
def calculate_PESQ(gen_output, sources):
    pesq = PesqLoss(0.5, sample_rate=44100, )
    mos = pesq.mos(sources, gen_output)
    loss = pesq(sources, gen_output)

    print(mos, loss)
    # loss.backward()
    nb_pesq = PerceptualEvaluationSpeechQuality(8000, 'nb')
    # print(f'Shapes: {gen_output.shape} {sources.shape}')
    # try: score = nb_pesq(gen_output, sources)
    # except:
    #     print(f'---------------------------------Except block reached------------------------------------')
    #     score = 0.000001
    # if score == 0: score = 1e-8
    # elif score < 0: score = (score + 0.5) / 5
    return score


def train(data_loader, model_gen, model_discr, optimizer_gen, optimizer_discr, loss_fn_discr, scheduler_gen, device, epochs=2):
    model_gen.train()
    model_discr.train()
    print('TRAINING')
    epochs = 2
    for epoch in range(epochs):
        factor = get_factor(epoch)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'EPOCH: {epoch + 1}')
        print(f'Current Time: {current_time}')
        start = time.time()
        for i, batch in enumerate(data_loader):
            # 1. Get the data
            # batch_size = 4
            mixture, lens, sources = batch
            mixture = mixture.unsqueeze(1)
            if i == 0:
                print(mixture.shape)
            # if mixture.size(0) < batch_size:
            #     mixture, sources, lens = pad_zeroes(mixture, sources, lens, batch_size)
            mixture = mixture.to(device)
            sources = sources.to(device)

            # # 2. Zero gradient
            optimizer_gen.zero_grad()
            # optimizer_discr.zero_grad()

# """A ----------------------------------------------CONCAT METHOD ---------------------------------------------------------------"""
# GEN o/p size: ([batch, 2, 32000])     DISCR o/p size: ([batch, 2, 64000])

# Gen_loss: loss_snr
# Discr_loss: COMPARE- discr(cat(sources, outputs_gen/est_sources)) to cat(sources_size_labels1, est_sources_size_labels0): ([batch, 2, 64000])
# loss_total = Gen_loss + factor * Discr_loss
# Backpropagate loss_total for both Optimizers (gens and discr)

            # # GENERATOR
            # # 3. Forward pass
            # outputs_gen = model_gen(mixture)
            #
            # # DISCRIMINATOR
            # # *. Prepare the data
            # input_discr = torch.cat((sources, outputs_gen), dim=2)
            # clean_labels = torch.ones((1, sources.size(2)), dtype=torch.float32)
            # estimated_labels = torch.zeros((1, outputs_gen.size(2)), dtype=torch.float32)
            # discr_labels = torch.cat((clean_labels, estimated_labels), dim=1).to(device)
            #
            # # 3. Forward pass
            # output_discr = model_discr(input_discr)
            # output_discr = torch.nn.functional.pad(output_discr, (0, 4))
            #
            # # 4. LOSS BACKWARDS BOTH
            # loss_gen, max_snr, estimate_source, reorder_estimate_source = cal_loss(outputs_gen, sources, lens, device)
            # loss_discr = loss_fn_discr(output_discr, discr_labels)
            # total_loss = loss_gen.float() + (0.01 * loss_discr.float())
            # total_loss.backward()
            #
            # # 5. Optimize
            # optimizer_gen.step()
            # optimizer_discr.step()
# """A ---------------------------------------------------------------------------------------------------------------------------"""

# """B ----------------------------------------------ORIGINAL METHOD -------------------------------------------------------------"""
# GEN o/p size: ([batch, 2, 32000])      DISCR o/p size: ([batch, 2, 8000/64000]) {3*downsampled or 2*upsampled}

# Gen_loss: loss_snr     Adversarial_loss: COMPARE- discr(outputs_gen/est_sources) to torch.full(discr_op_size_1): ([batch, 2, 8000/64000])
# Gen_loss_total: Gen_loss + factor * Adversarial_loss
# Backpropagate Gen_loss_total for Gen optimizer

# Discr_real_loss: COMPARE discr(sources) to torch.full(discr_op_size_1)    Discr_fake_loss: COMPARE discr(est_sources) to torch.full(discr_op_size_0)
# Discr_loss_total = Avg(Discr_real_loss, Discr_fake_loss)
# Backpropagate Discr_loss_total for Discr Optimizer

            # GENERATOR
            # 3. Forward pass
            separated_speech = model_gen(mixture)

            # LSGAN training
            loss_gen, max_snr, estimate_source, reorder_estimate_source = cal_loss(separated_speech, sources, lens, device)
            total_loss_gen = loss_gen
            total_loss_gen.backward()
            optimizer_gen.step()

            # Separation loss (SI-SNR) for the ConvTasNet
            loss_gen, max_snr, estimate_source, reorder_estimate_source = cal_loss(separated_speech, sources, lens, device)
            # Adversarial loss for ConvTasNet
            adversarial_op = model_discr(separated_speech)
            if i == 0:
                # print(f'Adversarial/Discr op shape: {adversarial_op.shape}\t Adversarial op: {adversarial_op}')
                print(f'Adversarial/Discr op shape: {adversarial_op.shape}')
            adversarial_loss = loss_fn_discr(adversarial_op, torch.ones_like(adversarial_op))
            # adversarial_loss = loss_fn_discr(adversarial_op, torch.full(adversarial_op.size(), 0.9).to(device))

            # 4. Total_Loss Backward for the Gen (ConvTasNet)
            total_loss_gen = loss_gen + (factor * adversarial_loss)
            total_loss_gen.backward()
            # total_loss_gen.backward(retain_graph=True)

            # 5. Optimize Gen
            optimizer_gen.step()

            # DISCRIMINATOR
            # 3. Forward pass
            real_output = model_discr(sources)
            outputs_gen = model_gen(mixture).detach()
            fake_output = model_discr(outputs_gen)

            # 4. Total_Loss Backward for the Discr
            loss_real = loss_fn_discr(real_output, torch.ones_like(real_output))
            loss_fake = loss_fn_discr(fake_output, torch.zeros_like(fake_output))
            # loss_real = loss_fn_discr(real_output, torch.full(real_output.size(), 0.9).to(device))
            # loss_fake = loss_fn_discr(fake_output, torch.full(fake_output.size(), 0.1).to(device))
            total_loss_discr = 0.5 * (loss_real + loss_fake)
            total_loss_discr.backward()

            # 5. Optimize Discr
            optimizer_discr.step()
# """B -----------------------------------------------------------------------------------------------------------------------------"""

# """C -----------------------------------------------PESQ METHOD ------------------------------------------------------------------"""
# GEN o/p size: ([batch, 2, 32000])      DISCR o/p size: ([1])

# Gen_loss: loss_snr     Adversarial_loss: COMPARE- discr(cat(outputs_gen/est_sources, sources)) to 1
# Gen_loss_total: Gen_loss + factor * Adversarial_loss
# Backpropagate Gen_loss_total for Gen optimizer

# Discr_PESQ_loss: COMPARE discr(torch.cat(outputs_gen/est_sources, sources)) to PESQ score     Discr_source_loss: COMPARE discr(torch.cat(sources, sources)) to 1
# Discr_loss_total = Discr_PESQ_loss + Discr_source_loss
# Backpropagate Discr_loss_total for Discr Optimizer

            # # GENERATOR
            # # 3. Forward pass
            # separated_speech = model_gen(mixture)
            #
            # # *. PESQ score
            # PESQ = calculate_PESQ(separated_speech, sources)
            # # *. Separation loss (SI-SNR) for the ConvTasNet
            # loss_gen, max_snr, estimate_source, reorder_estimate_source = cal_loss(separated_speech, sources, lens, device)
            # # *. Adversarial loss for ConvTasNet-PESQ
            # input_discr = torch.cat((separated_speech, sources), dim=2)
            # adversarial_op = model_discr(input_discr)
            # if i < 2:
            #     print(f'Adversarial/Discr op shape: {adversarial_op.shape}\t Adversarial op: {adversarial_op}')
            # adversarial_loss = (adversarial_op - 1)**2
            #
            # # 4. Total_Loss Backward for the Gen (ConvTasNet)
            # total_loss_gen = loss_gen + (factor * adversarial_loss)
            # total_loss_gen.backward()
            # # total_loss_gen.backward(retain_graph=True)
            #
            # # 5. Optimize Gen
            # optimizer_gen.step()
            #
            # # DISCRIMINATOR
            # # 3. Forward pass
            # outputs_gen = model_gen(mixture).detach()
            # input1_discr = torch.cat((outputs_gen, sources), dim=2)
            # input2_discr = torch.cat((sources, sources), dim=2)
            # output1_discr = model_discr(input1_discr)
            # output2_discr = model_discr(input2_discr)
            #
            # # 4. Total_Loss Backward for the Discr
            # discr_pesq_loss = (output1_discr - PESQ)**2
            # discr_source_loss = (output2_discr - 1)**2
            # total_loss_discr = discr_pesq_loss + discr_source_loss
            # total_loss_discr.backward()
            #
            # # 5. Optimize Discr
            # optimizer_discr.step()
# """C -----------------------------------------------------------------------------------------------------------------------------"""

            a = 264
            if (i + 1) % a == 0:
                total = 0
                training_time = (((time.time() - start) / 60) * (1 / a) * ((len(data_loader) - (i + 1)) + (len(data_loader)) * (epochs - (epoch + 1))))
                total_hrs, total_mins, final_hr, final_min = training_time_calc(training_time)
                for r in range(max_snr.shape[0]):
                    total += max_snr[r].item()
                avg_snr = total / max_snr.shape[0]
                print(f'Epoch: {epoch + 1}\tBatch: {i + 1}')
                print(f'\ttime/batch: {(time.time() - start) / 60:.3f}m per {a} data\tTime left: {total_hrs} hours {total_mins:.1f} mins\tEst. finish time: {final_hr}: {final_min}')
                print(f'\tSNR: {avg_snr:.3f}')
                print(f'\tGEN LOSS: snr_loss: {loss_gen.item():.5f}\tadversarial_loss: {adversarial_loss.item():.5f}\ttotal_gen_loss: {total_loss_gen.item():.5f}')
                print(f'\tDISCR LOSS: real_loss: {loss_real:.5f}\tfake_loss: {loss_fake:.5f}\ttotal_discr_loss: {total_loss_discr.item():.5f}')
                # print(f'\tPESQ: {PESQ:.4f}\toutput1_discr: {output1_discr.item():.4f}\tdiscr_pesq_loss: {discr_pesq_loss.item():.4f}\toutput2_discr: {output2_discr.item():.4f}\tdiscr_source_loss: {discr_source_loss.item():.4f}\ttotal_discr_loss: {total_loss_discr.item():.5f}')
                start = time.time()
            # if i + 1  == len(data_loader):
            #     perform_validation_check(model_gen)

        scheduler_gen.step()
    print('Training Finished')
    torch.save(model_gen.state_dict(),
               '/ConvTasNet_PyTorch/WSJ0-2Mix/WSJ0-2Mix_Saved_models/2.1.model-b24-GAN2.1-10epochs.pth')
