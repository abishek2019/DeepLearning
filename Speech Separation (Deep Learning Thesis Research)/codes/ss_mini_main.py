import sys

import torch
from Models_Luo import TasNet
from Test_and_Predict import test, predict
from Data_load_preprocess import preprocess, AudioDataset, AudioDataLoader
from datasets_path import datasets_location
if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
# ---------------------------------------------Change here---------------------------------------------------------------------------------
    # Hyperparameters
    batch_size = 32
    sample_rate = 8000
    json_prep = False
    find_sdr = False
    tt_segment = -1
    dataset_names = ['Libri2Mix_8k_min_100_clean', 'Libri2Mix_8k_min_360_clean', 'Libri2Mix_16k_min_100_clean', 'VCTK2MIX_16k_min_clean', 'WSJ02Mix_min']

    current_dataset = dataset_names[0]
    tr_location, ev_location, tt_location = datasets_location(current_dataset)
    nonGAN_path = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Saved_Models/Libri-2Mix_Saved_models/8k_min_100_clean/NonGAN/B32_g16k_s4/New1_NonGAN_g16k_s4_e100_best.pth'
    vanillaGAN_path = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Saved_Models/Libri-2Mix_Saved_models/8k_min_100_clean/LSGAN/LSGAN_g16k_s4_e100_best.pth'
    lsGAN_path = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Saved_Models/Libri-2Mix_Saved_models/8k_min_100_clean/LSGAN/LSGAN_g16k_s4_e100_best.pth'
    metricGAN_path = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Saved_Models/Libri-2Mix_Saved_models/8k_min_100_clean/LSGAN/LSGAN_g16k_s4_e100_best.pth'
# -------------------------------------------------------------------------------------------------------------------------------------

    # 1 LOAD AND PREPROCESS THE DATA
    if json_prep:
        # Preprocess 3.3.test Data
        args_tt = {'in_dir': tt_location, 'out_dir': tt_location, 'sample_rate': sample_rate}
        preprocess(args_tt)

    # Load 3.3.test Data
    json_dir_tt = tt_location + '/o-p'
    dataset_tt = AudioDataset(json_dir_tt, batch_size=batch_size, sample_rate=sample_rate, segment=tt_segment)
    dataloader_tt = AudioDataLoader(dataset_tt, batch_size=1, num_workers=2)

    print(f'DatasetName: {current_dataset}\tDataLoader len: {len(dataloader_tt)}')

    # Load Models
    model_gen1, model_gen2, model_gen3, model_gen4 = TasNet().to(device), TasNet().to(device), TasNet().to(device), TasNet().to(device)

    model_gen1.load_state_dict(torch.load(nonGAN_path))
    # model_gen2.load_state_dict(torch.load(vanillaGAN_path))
    # model_gen3.load_state_dict(torch.load(lsGAN_path))
    # model_gen4.load_state_dict(torch.load(metricGAN_path))

    # 3.3.test Si-SNRi and SDRi of models
    model_gen, model_type = None, None
    for i in range(1, 5):
        if i == 1:
            model_gen, model_type = model_gen1, 'NONGAN'
        elif i == 2:
            model_gen, model_type = model_gen2, '\nVanilla GAN'
        elif i == 3:
            model_gen, model_type = model_gen3, '\nLSGAN'
        elif i == 4:
            model_gen, model_type = model_gen4, '\nMetricGAN'

        args_test = {'device': device, 'dataloader': dataloader_tt, 'model_gen': model_gen, 'find_sdr': find_sdr}
        print(model_type)
        test(args_test)


    # # predict
    # predict_file = '/home/abishek/speech_proj/ConvTasNet_PyTorch/WSJ0-2Mix/1.train/Data_train/Mix/01aa010i_4.7564_209a010v_-4.7564.wav'
    # i_p_shape, o_p = predict(predict_file, model_gen, device)
    # print(f'Predict audio shape: {i_p_shape}')
    # print(f'Prediction Output: {o_p}')

    # g_snr_losses = [1, 2]
    # g_adv_losses = [1]
    # g_total_losses = [1]
    # d_real_losses = [1]
    # d_fake_losses = [1]
    # d_total_losses = [1]
    # losses_list = [g_snr_losses, g_adv_losses, g_total_losses, d_real_losses, d_fake_losses, d_total_losses]
    # losses_names = ['g_snr_losses', 'g_adv_losses', 'g_total_losses', 'd_real_losses', 'd_fake_losses',
    #                 'd_total_losses']
    # for i, my_list in enumerate(losses_list):
    #     file_name = f'/{losses_names[i]}.txt'
    #     with open('/home/abishek/speech_proj/ConvTasNet_PyTorch/Losses_txt/LSGAN' + file_name, 'w') as file:
    #         for item in my_list:
    #             file.write(f"{item}\n")
    #
    # # Read the file into a list
    # with open('/home/abishek/speech_proj/ConvTasNet_PyTorch/Losses_txt/LSGAN/g_snr_losses.txt', 'r') as file:
    #     file_content = file.readlines()
    #
    # # Display the content of the list
    # file_content_stripped = [line.strip() for line in file_content]
    # print(file_content_stripped)
    # print(type(file_content_stripped))
    #
    # sys.exit()