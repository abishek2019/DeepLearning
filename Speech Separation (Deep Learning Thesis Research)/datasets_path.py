def datasets_location(name):
    tr, ev, tt = None, None, None
    # Libri2Mix_8k_min_100_clean
    if name == 'Libri2Mix_8k_min_100_clean':
        tr = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Libri2Mix/8k_min_100_clean/1.train'
        ev = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Libri2Mix/8k_min_100_clean/2.validation'
        tt = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Libri2Mix/8k_min_100_clean/3.test'

    # Libri2Mix_8k_min_360_clean
    elif name == 'Libri2Mix_8k_min_360_clean':
        tr = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Audio_Data/Libri-2Mix/8k_min_360_clean/1.train'
        ev = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Audio_Data/Libri-2Mix/8k_min_360_clean/2.validation'
        tt = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Audio_Data/Libri-2Mix/8k_min_360_clean/3.3.test'

    # Libri2Mix_16k_min_100_clean
    elif name == 'Libri2Mix_16k_min_100_clean':
        tr = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Audio_Data/Libri-2Mix/16k_min_100_clean/1.train'
        ev = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Audio_Data/Libri-2Mix/16k_min_100_clean/2.validation'
        tt = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Audio_Data/Libri-2Mix/16k_min_100_clean/3.3.test'

    # VCTK2MIX_16k_min_clean
    elif name == 'VCTK2MIX_16k_min_clean':
        tt = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Audio_Data/VCTK-2Mix/Clean_16k_min/3.test'

    # WSJ02Mix_min
    elif name == 'WSJ02Mix_min':
        tr = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Audio_Data/WSJ0-2Mix/1.train'
        ev = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Audio_Data/WSJ0-2Mix/2.validation'
        tt = '/home/abishek/speech_proj/ConvTasNet_PyTorch/Data/Audio_Data/WSJ0-2Mix/3.3.test'

    return tr, ev, tt


