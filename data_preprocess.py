import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='EDM for kol')
parser.add_argument('--data_path', type=str, default='data/kf_2d_re1000_256_40seed.npy', help='your path for saving the dataset')
config = parser.parse_args()


def main():
    data_path = config.data_path
    coarse_dim = 32
    down_sampler = torch.nn.Upsample(size=coarse_dim, mode='bilinear')

    data = np.load(data_path)
    data = torch.from_numpy(data)

    latent_data = down_sampler(data)
    # print(latent_data.shape)

    np.save('data/kf_2d_re1000_256_40seed_train.npy', data[:36])
    np.save('data/kf_2d_re1000_256_40seed_valid.npy', data[36:])

    np.save('data/kf_2d_re1000_32_40seed_bthwc_train.npy', latent_data[:36])
    np.save('data/kf_2d_re1000_32_40seed_bthwc_valid.npy', latent_data[36:])

if __name__ == "__main__":
    main()
