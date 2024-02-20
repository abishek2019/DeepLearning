import torch
from Models import ConvTasNet
from Models_new import Discriminator
from Models_Luo import TasNet
from Train_new import train_new
from Test_and_Predict import test
from torch.optim import lr_scheduler
from Data_load_preprocess import preprocess, AudioDataset, AudioDataLoader
from datasets_path import datasets_location

# Execution
if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

# --------------------------Change here only--------------------------------------------------------------------------------------------
    # Hyperparameters
    mode = 'Baseline'             # modes: Baseline, VanillaGAN, LSGAN, MetricGAN
    batch_size = 32
    sample_rate = 8000
    lr_gen = 0.001
    lr_discr = 0.0005
    adversarial_factor = 10

    # Misc
    discr_in_ch = 2
    if mode == 'MetricGAN':
        discr_in_ch = 4
    json_prep = False
    find_sdr = False
    smoothing = 'No'
    tr_segment, ev_segment, tt_segment = 4, -1, -1

    dataset_names = ['Libri2Mix_8k_min_100_clean', 'Libri2Mix_8k_min_360_clean', 'Libri2Mix_16k_min_100_clean', 'VCTK2MIX_16k_min_clean', 'WSJ02Mix_min']

    current_dataset = dataset_names[0]
    tr_location, ev_location, tt_location = datasets_location(current_dataset)
# -------------------------------------------------------------------------------------------------------------------------------------

    # 1 LOAD AND PREPROCESS THE DATA
    if json_prep:
        # Preprocess 1.train Data
        args_tr = {'in_dir': tr_location, 'out_dir': tr_location, 'sample_rate': sample_rate}
        preprocess(args_tr)
        # Preprocess 2.validation Data
        args_ev = {'in_dir': ev_location, 'out_dir': ev_location, 'sample_rate': sample_rate}
        preprocess(args_ev)
        # Preprocess 3.3.test Data
        args_tt = {'in_dir': tt_location, 'out_dir': tt_location, 'sample_rate': sample_rate}
        preprocess(args_tt)

    # Load 1.train Data
    json_dir_tr = tr_location + '/o-p'
    dataset_tr = AudioDataset(json_dir_tr, batch_size = batch_size, sample_rate=sample_rate, segment=tr_segment)
    dataloader_tr = AudioDataLoader(dataset_tr, batch_size=1, num_workers=2)
    # Load 2.validation Data
    json_dir_ev = ev_location + '/o-p'
    dataset_ev = AudioDataset(json_dir_ev, batch_size = batch_size, sample_rate=sample_rate, segment=ev_segment)
    dataloader_ev = AudioDataLoader(dataset_ev, batch_size=1, num_workers=2)
    # Load 3.3.test Data
    json_dir_tt = tt_location + '/o-p'
    dataset_tt = AudioDataset(json_dir_tt, batch_size = batch_size, sample_rate=sample_rate, segment=tt_segment)
    dataloader_tt = AudioDataLoader(dataset_tt, batch_size=1, num_workers=2)

    # 2 Define MODEL, 3 LOSS FN and OPTIMIZER
    model_gen = TasNet().to(device)
    # model_gen = ConvTasNet(2).to(device)
    # model_gen.load_state_dict(torch.load('/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Saved_Models/Libri-2Mix_Saved_models/8k_NonGAN_e100_best.pth'))
    model_discr, optimizer_discr, loss_fn_discr = None, None, None
    # Gen loss fns and opt
    optimizer_gen = torch.optim.Adam(model_gen.parameters(), lr=lr_gen, weight_decay=0.0001)
    scheduler_gen = lr_scheduler.ReduceLROnPlateau(optimizer_gen, mode='max', patience=3, factor=0.5, verbose=True)
    if mode != 'Baseline':
        model_discr = Discriminator(discr_in_ch).to(device)
        # Discr loss fns and opt
        optimizer_discr = torch.optim.Adam(model_discr.parameters(), lr=lr_discr)
        loss_fn_discr = torch.nn.MSELoss()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'DatasetName: {current_dataset}\tDataloader_tr: {len(dataloader_tr)}\tDataloader_ev: {len(dataloader_ev)}\tDataloader_tt: {len(dataloader_tt)}')
    print(f'Mode: {mode}')
    print(f'Batch_size: {batch_size}\tSample Rate: {sample_rate}')
    print(f'Generator params total: {count_parameters(model_gen)}\tLR Gen: {lr_gen}')
    if mode != 'Baseline':
        print(f'Discriminator params total: {count_parameters(model_discr)}\tLR Discr: {lr_discr}')
        print(f'Network params total: {count_parameters(model_gen) + count_parameters(model_discr)}')
        print(f'Factor: {adversarial_factor}. Smoothing: {smoothing}')

    args_train = {'mode': mode,
                  'device': device,
                  'dataloader': dataloader_tr,
                  'dataloader_eval': dataloader_ev,
                  'model_gen': model_gen,
                  'optimizer_gen': optimizer_gen,
                  'scheduler_gen': scheduler_gen,
                  'model_discr': model_discr,
                  'optimizer_discr': optimizer_discr,
                  'loss_fn_discr': loss_fn_discr,
                  'adversarial_factor': adversarial_factor
                  }

    # 4 TRAIN THE MODEL
    train_new(args_train)

    # 5 TEST THE MODEL
    args_test = {'device': device, 'dataloader': dataloader_tt, 'model_gen': model_gen, 'find_sdr': find_sdr}
    test(args_test)

    # # 6 PREDICT
    # for data, *_ in dataloader_tr:
    #     break
    # with torch.no_grad():
    #     data = data.unsqueeze(1)
    #     data = data.to(device)
    #     print(f'Prediction Shape: {data.shape}')
    #     output = model_gen(data)
    #     # print(output)
