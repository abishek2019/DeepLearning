import torch
import torchaudio
from Data_load_preprocess import printf
from Calculate_loss import cal_loss, cal_SISNRi, cal_SDRi
from Data_load_preprocess import remove_pad
import math


# 5 TEST THE MODEL
def test(args, val_check = False):
    device = args['device']
    dataloader = args['dataloader']
    model_gen = args['model_gen']
    find_sdr = args['find_sdr']

    total_SISNRi, total_SDRi, total_cnt = 0, 0, 0

    model_gen.eval()
    with torch.no_grad():
        for i, (mixture, lens, sources) in enumerate(dataloader):
            mixture = mixture.unsqueeze(1)
            mixture = mixture.to(device)
            sources = sources.to(device)
            estimate_source = model_gen(mixture)
            loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(sources, estimate_source, lens, device)
            # Remove padding and flat
            mixture = remove_pad(mixture, lens)
            sources = remove_pad(sources, lens)
            # NOTE: use reorder estimate source
            estimate_source = remove_pad(reorder_estimate_source,  lens)

            count, batch_SISNRi = 0, 0
            # for each utterance
            for mix, src_ref, src_est in zip(mixture, sources, estimate_source):
                # Compute SDRi
                if find_sdr:
                    avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                    total_SDRi += avg_SDRi

                # Compute SI-SNRi
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix[0])
                total_SISNRi += avg_SISNRi
                batch_SISNRi += avg_SISNRi
                total_cnt += 1
                count += 1

            if not val_check:
                if (i + 1) % 15 == 0:
                    print(f'Batch: {i + 1}\tAvg_Si-SNRi: {batch_SISNRi / count:.2f}')
                    # print("Avg Si-SNRi Batch:", i + 1, round(batch_SISNRi / count, 2))

    SISNRi = round(total_SISNRi / total_cnt, 2)
    print(f'Average Si-SNR Improvement: {SISNRi:.2f}')
    if find_sdr:
        SDRi = round(total_SDRi / total_cnt, 2)
        print(f'Average SDR Improvement: {SDRi:.2f}')
    return SISNRi

# 6 PREDICT
def predict(predict_file, model_gen, device):
    # Define and apply effects
    # def apply_effect(waveform, sample_rate):
    #     effect = ",".join(["lowpass=frequency=300:poles=1",  # apply single-pole lowpass filter
    #                        "atempo=0.8",  # reduce the speed
    #                        "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|delays=400:decays=0.3"
    #                        # Applying echo gives some dramatic feeling
    #                        ])
    #     effector = torchaudio.io.AudioEffector(effect=effect)
    #     return effector.apply(waveform, sample_rate)
    input, sr = torchaudio.load(predict_file)
    with torch.no_grad():
        input = input.unsqueeze(1).to(device)
        output = model_gen(input)
        return input.shape, output
    # S1, S2 = apply_effect(output[0][0], sr), apply_effect(output[0][1], sr)
    # noise_suppressed_s1, noise_suppressed_s2 = librosa.denoise(output[0][0]), librosa.denoise(output[0][1])
    # display(Audio(data[0].cpu().numpy(), rate=sr))
    # display(Audio(output[0][0].cpu().numpy(), rate=sr))
    # display(Audio(output[0][1].cpu().numpy(), rate=sr))

    # # save the predicted wav files
    # path = '/speech_proj/ConvTasNet_PyTorch/WSJ0-2Mix/6.Separated_audios'
    # output = output.to('cpu')
    # S1, S2 = output[0][0], output[0][1]
    # S1, S2 = S1.unsqueeze(0), S2.unsqueeze(0)
    # print(S2.shape)
    # # torchaudio.save(path + '/S1.wav', S1, 8000)
    # torchaudio.save(path+ '/S2.wav', S2, 8000)





