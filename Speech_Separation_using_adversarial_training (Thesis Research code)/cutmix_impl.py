import torch
from Data_load_preprocess import preprocess, AudioDataset, AudioDataLoader

def cutmix(mixture, sources, alpha=1.0):
    batch_size = mixture.size(0)

    # Generate random indices for mixing
    rand_indices = torch.randperm(batch_size)
    target_a, target_b = sources, sources[rand_indices]

    # Generate binary mask
    lam = torch.tensor(alpha)  # Set to alpha for consistent API
    binary_mask = (torch.rand(batch_size) < lam).float().view(-1, 1, 1).to(device)

    # Apply CutMix using the binary mask
    new_mixed_data = mixture.clone()
    new_mixed_data[:, :, :] = mixture[:, :, :] * binary_mask + mixture[rand_indices, :, :] * (1 - binary_mask)

    # Adjust targets based on the mixing ratio
    mixed_sources = sources * binary_mask + sources[rand_indices] * (1 - binary_mask)

    return new_mixed_data, mixed_sources, binary_mask.squeeze()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 3.3.test
    # in_dir = '/home/abishek/speech_proj/ConvTasNet_PyTorch/WSJ0-2Mix/3.3.test/Data_test'
    # out_dir = '/home/abishek/speech_proj/ConvTasNet_PyTorch/WSJ0-2Mix/3.3.test/Data_test-O-P'
    # args = {'in_dir': in_dir, 'out_dir': out_dir, 'sample_rate': 8000}
    # preprocess(args)
    json_dir = '/ConvTasNet_PyTorch/WSJ0-2Mix/3.3.test/Data_test-O-P/o-p'
    test_data = AudioDataset(json_dir, 1)
    test_dataLoader = AudioDataLoader(test_data, batch_size=1, num_workers=2)
    for i, (mixture, lens, sources) in enumerate(test_dataLoader):
        if i == 0:
            mixture = mixture.unsqueeze(1)
            mixture = mixture.to(device)
            sources = sources.to(device)
            mixed_data, mixed_sources, binary_mask = cutmix(mixture, sources)
            print(f'Mixed data: {mixed_data.shape} Mixed sources: {mixed_sources.shape} Binary Mask: {binary_mask.shape}')
        break
