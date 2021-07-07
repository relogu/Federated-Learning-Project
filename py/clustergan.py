# from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cluster_gan/clustergan.py
from __future__ import print_function

try:
    import sys
    import pathlib
    import argparse
    import os
    import numpy as np

    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    
    from itertools import chain as ichain
    
    path = pathlib.Path(__file__).parent.absolute()
    sys.path.append(str(path.parent))
    
    import clustering.py.common_fn as my_fn
    import py.metrics as my_metrics
    import py.dataset_util as data_util

except ImportError as e:
    print(e)
    raise ImportError

network_setup_string = "Setting up {}...\n"

def get_parser():
    parser = argparse.ArgumentParser(description="ClusterGAN Training Script")
    parser.add_argument("-s", "--dataset", dest="dataset", default='mnist', choices=['mnist', 'euromds'], type=type(''), help="Dataset")
    parser.add_argument("--save_img", dest="save_img", default=False, type=bool, help="Wheather to save images")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("-i", "--img_size", dest="img_size", type=int, default=28, help="Size of image dimension")
    parser.add_argument("-d", "--latent_dim", dest="latent_dim", default=30, type=int, help="Dimension of latent space")
    parser.add_argument("-l", "--lr", dest="learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-c", "--n_critic", dest="n_critic", type=int, default=5, help="Number of training steps for discriminator per iter")
    parser.add_argument("-w", "--wass_flag", dest="wass_flag", action='store_true', help="Flag for Wasserstein metric")
    parser.add_argument("-a", "--hardware_acc", dest="cuda_flag", action='store_true', help="Flag for hardware acceleration using cuda (if available)")
    parser.add_argument("-f", "--folder", dest="out_folder", type= type(str('')), help="Folder to output images")
    parser.add_argument('-g', '--groups', dest='groups',
                        required=False,
                        type=int,
                        choices=[1,2,3,4,5,6,7],
                        default=7,
                        action='store',
                        help='how many groups of variables to use for EUROMDS dataset')
    return parser

# Sample a random latent space vector
def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False, cuda=False):

    assert (fix_class == -1 or (fix_class >= 0 and fix_class < n_c) ), "Requested class %i outside bounds."%fix_class
    TENSOR = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Sample noise as generator input, zn
    zn = Variable(TENSOR(0.75*np.random.normal(0, 1, (shape, latent_dim))), requires_grad=req_grad)

    ######### zc, zc_idx variables with grads, and zc to one-hot vector
    # Pure one-hot vector generation
    zc_ft = TENSOR(shape, n_c).fill_(0)
    zc_idx = torch.empty(shape, dtype=torch.long)

    if (fix_class == -1):
        zc_idx = zc_idx.random_(n_c).cuda() if cuda else zc_idx.random_(n_c)
        zc_ft = zc_ft.scatter_(1, zc_idx.unsqueeze(1), 1.)
    else:
        zc_idx[:] = fix_class
        zc_ft[:, fix_class] = 1

        if cuda:
            zc_idx = zc_idx.cuda()
            zc_ft = zc_ft.cuda()

    zc = Variable(zc_ft, requires_grad=req_grad)

    # Return components of latent space variable
    return zn, zc, zc_idx


def calc_gradient_penalty(net_d, real_data, generated_data, cuda=False):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    if cuda: 
        alpha = alpha.cuda()
    
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    if cuda: 
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = net_d(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda() if cuda else torch.ones(prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) \
            or isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


# Softmax function
def softmax(x):
    return F.softmax(x, dim=1)


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """
    def __init__(self, shape = None):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
    def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'shape={}'.format(
                self.shape
            )


class GeneratorCNN(nn.Module):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, latent_dim, n_c, gen_dims, x_shape, verbose=False):
        super(GeneratorCNN, self).__init__()

        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.gen_dims = gen_dims
        self.x_shape = x_shape
        self.verbose = verbose
        
        self.model = nn.Sequential(
            # Fully connected layers
            torch.nn.Linear(self.latent_dim + self.n_c, self.gen_dims[0]),
            nn.BatchNorm1d(self.gen_dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.gen_dims[0], self.gen_dims[1]),
            nn.BatchNorm1d(self.gen_dims[1]),
            nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(self.gen_dims[1], self.gen_dims[2]),
            nn.BatchNorm1d(self.gen_dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            
            torch.nn.Linear(self.gen_dims[2], self.gen_dims[3]),
            nn.Sigmoid()
        )

        initialize_weights(self)

        if self.verbose:
            print(network_setup_string.format(self.name))
            print(self.model)
    
    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_gen = self.model(z)
        # Reshape for output
        x_gen = x_gen.view(x_gen.size(0), self.x_shape)
        return x_gen


class ConvGeneratorCNN(nn.Module):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, latent_dim, n_c, x_shape, verbose=False):
        super(ConvGeneratorCNN, self).__init__()

        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = (128, 7, 7)
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose
        
        self.model = nn.Sequential(
            # Fully connected layers
            torch.nn.Linear(self.latent_dim + self.n_c, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.LeakyReLU(0.2, inplace=True),
        
            # Reshape to 128 x (7x7)
            Reshape(self.ishape),

            # Upconvolution layers
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
        )

        initialize_weights(self)

        if self.verbose:
            print(network_setup_string.format(self.name))
            print(self.model)
    
    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_gen = self.model(z)
        # Reshape for output
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen


class EncoderCNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, latent_dim, enc_dims, n_c, verbose=False):
        super(EncoderCNN, self).__init__()

        self.name = 'encoder'
        self.latent_dim = latent_dim
        self.enc_dims = enc_dims
        self.n_c = n_c
        self.verbose = verbose
        
        self.model = nn.Sequential(
            # Fully connected layers
            torch.nn.Linear(self.enc_dims[0], self.enc_dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.enc_dims[1], self.enc_dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.enc_dims[2], self.enc_dims[3]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.enc_dims[3], latent_dim + n_c)
        )

        initialize_weights(self)
        
        if self.verbose:
            print(network_setup_string.format(self.name))
            print(self.model)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)
        # continuous components
        zn = z[:, 0:self.latent_dim]
        # one-hot components
        zc_logits = z[:, self.latent_dim:]
        # Softmax on one-hot component
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class ConvEncoderCNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, latent_dim, n_c, verbose=False):
        super(ConvEncoderCNN, self).__init__()

        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose
        
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Flatten
            Reshape(self.lshape),
            
            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, latent_dim + n_c)
        )

        initialize_weights(self)
        
        if self.verbose:
            print(network_setup_string.format(self.name))
            print(self.model)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)
        # continuous components
        zn = z[:, 0:self.latent_dim]
        # one-hot components
        zc_logits = z[:, self.latent_dim:]
        # Softmax on one-hot component
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class DiscriminatorCNN(nn.Module):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    Output is a 1-dimensional value
    """            
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, disc_dims, wass_metric=False, verbose=False):
        super(DiscriminatorCNN, self).__init__()
        
        self.name = 'discriminator'
        self.disc_dims = disc_dims
        self.wass = wass_metric
        self.verbose = verbose
        
        self.model = nn.Sequential(
            # Fully connected layers
            torch.nn.Linear(self.disc_dims[0], self.disc_dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.disc_dims[1], self.disc_dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.disc_dims[2], self.disc_dims[3]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(self.disc_dims[3], 1)
        )
        
        # If NOT using Wasserstein metric, final Sigmoid
        if (not self.wass):
            self.model = nn.Sequential(self.model, torch.nn.Sigmoid())
        
        initialize_weights(self)
        
        if self.verbose:
            print(network_setup_string.format(self.name))
            print(self.model)

    def forward(self, img):
        # Get output
        validity = self.model(img)
        return validity
    

class ConvDiscriminatorCNN(nn.Module):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    Output is a 1-dimensional value
    """            
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, wass_metric=False, verbose=False):
        super(ConvDiscriminatorCNN, self).__init__()
        
        self.name = 'discriminator'
        self.channels = 1
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = wass_metric
        self.verbose = verbose
        
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Flatten
            Reshape(self.lshape),
            
            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, 1),
        )
        
        # If NOT using Wasserstein metric, final Sigmoid
        if (not self.wass):
            self.model = nn.Sequential(self.model, torch.nn.Sigmoid())

        initialize_weights(self)

        if self.verbose:
            print(network_setup_string.format(self.name))
            print(self.model)

    def forward(self, img):
        # Get output
        validity = self.model(img)
        return validity

if __name__ == "__main__":
    # get parameters
    args = get_parser().parse_args()
    
    # defining output folder
    if args.out_folder is None:
        path_to_out = pathlib.Path(__file__).parent.parent.absolute()
    else:
        path_to_out = args.out_folder
    
    os.makedirs(path_to_out, exist_ok=True)

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    test_batch_size = 5000
    lr = args.learning_rate
    b1 = 0.5
    b2 = 0.9
    decay = 2.5*1e-5
    n_skip_iter = args.n_critic
    
    # Latent space info
    latent_dim = args.latent_dim
    n_c = 10
    betan = 10
    betac = 10

    # Wasserstein+GP metric flag
    wass_metric = args.wass_flag
    print('Using metric {}'.format('Wassestrain' if wass_metric else 'Vanilla'))

    CUDA = True if (torch.cuda.is_available() and args.cuda_flag) else False
    device = torch.device('cuda:0' if CUDA else 'cpu')
    print('Using device {}'.format(device))

    # Data dimensions
    if args.dataset == 'mnist':
        img_size = args.img_size
        channels = 1
        x_shape = (channels, img_size, img_size)
        
        # Initialize generator and discriminator
        generator = ConvGeneratorCNN(latent_dim, n_c, x_shape)
        encoder = ConvEncoderCNN(latent_dim, n_c)
        discriminator = ConvDiscriminatorCNN(wass_metric=wass_metric)
        
        # Configure data loader
        #os.makedirs("../../data/mnist", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../../data/mnist/",
                train=True,
                transform=transforms.Compose(
                    [transforms.ToTensor()]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        # Test data loader
        testdata = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../../data/mnist/",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor()]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    
    else:
        groups = ['Genetics', 'CNA', 'Demographics', 'Clinical', 'GeneGene', 'CytoCyto', 'GeneCyto']
        # getting the entire dataset
        x = data_util.get_euromds_dataset(groups=groups[:args.groups])
        # getting labels from HDP
        prob = data_util.get_euromds_dataset(groups=['HDP'])
        y = []
        for label, row in prob.iterrows():
            if np.sum(row) > 0:
                y.append(row.argmax())
            else:
                y.append(-1)
        y = np.array(y)
        # getting the outcomes
        outcomes = data_util.get_outcome_euromds_dataset()
        # getting IDs
        ids = data_util.get_euromds_ids()
        n_features = len(x.columns)
        x = np.array(x)
        outcomes = np.array(outcomes)
        ids = np.array(ids)
        # cross-val
        train_idx, test_idx = data_util.split_dataset(
            x=x,
            splits=5,
            fold_n=0)
        # dividing data
        x_train = x[train_idx]
        y_train = y[train_idx]
        id_train = ids[train_idx]
        outcomes_train = outcomes[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]
        id_test = ids[test_idx]
        outcomes_test = outcomes[test_idx]
        dataloader = DataLoader(
            data_util.PrepareData(x=x_train,
                        y=y_train,
                        ids=id_train,
                        outcomes=outcomes_train),
            batch_size=batch_size)
        testloader = DataLoader(
            data_util.PrepareData(x=x_test,
                        y=y_test,
                        ids=id_test,
                        outcomes=outcomes_test),
            batch_size=batch_size)
        config = {
            'gen_dims': [int(4*n_features), int(3*n_features), int(2*n_features), x.shape[-1]],
            'enc_dims': [int(x.shape[-1]), int(4*n_features), int(3*n_features), int(2*n_features)],
            'disc_dims': [int(x.shape[-1]), int(2*n_features), int(3*n_features), int(4*n_features)]
        }
        generator = GeneratorCNN(latent_dim=latent_dim,
                                        n_c=n_c,
                                        gen_dims=config['gen_dims'],
                                        x_shape=x.shape[-1])
        encoder = EncoderCNN(latent_dim=latent_dim,
                                    enc_dims=config['enc_dims'],
                                    n_c=n_c)
        discriminator = DiscriminatorCNN(
            disc_dims=config['disc_dims'], wass_metric=wass_metric)

    torch.autograd.set_detect_anomaly(True)

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()


    if CUDA:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
        
    TENSOR = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    ge_chain = ichain(generator.parameters(),
                    encoder.parameters())

    optimizer_GE = torch.optim.Adam(ge_chain, lr=lr, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # ----------
    #  Training
    # ----------
    ge_l = []
    d_l = []

    c_zn = []
    c_zc = []
    c_i = []

    # Training loop 
    print('\nBegin training session with %i epochs...\n'%(n_epochs))
    for epoch in range(n_epochs):
        #for i, (imgs, itruth_label) in enumerate(dataloader):
        for i, (data) in enumerate(dataloader):
        
            if args.dataset == 'mnist':
                (imgs, itruth_label) = data
            elif args.dataset == 'euromds':
                (imgs, itruth_label, _, _) = data
            # Ensure generator/encoder are trainable
            generator.train()
            encoder.train()

            # Zero gradients for models, resetting at each iteration because they sum up,
            # and we don't want them to pile up between different iterations
            generator.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()
            optimizer_D.zero_grad()
            optimizer_GE.zero_grad()
            
            # Configure input
            real_imgs = Variable(imgs.type(TENSOR))

            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------
            
            # Sample random latent variables
            zn, zc, zc_idx = sample_z(shape=imgs.shape[0],
                                    latent_dim=latent_dim,
                                    n_c=n_c,
                                    cuda=CUDA)

            # Generate a batch of images
            gen_imgs = generator(zn, zc)
            
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs)
            valid = Variable(TENSOR(gen_imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(TENSOR(gen_imgs.size(0), 1).fill_(0.0), requires_grad=False)
            
            # Step for Generator & Encoder, n_skip_iter times less than for discriminator
            if (i % n_skip_iter == 0):
                # Encode the generated images
                enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)

                # Calculate losses for z_n, z_c
                zn_loss = mse_loss(enc_gen_zn, zn)
                zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)

                # Check requested metric
                if wass_metric:
                    # Wasserstein GAN loss
                    #ge_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss # original
                    ge_loss = - torch.mean(D_gen) + betan * zn_loss + betac * zc_loss # corrected
                else:
                    # Vanilla GAN loss
                    v_loss = bce_loss(D_gen, valid)
                    ge_loss = v_loss + betan * zn_loss + betac * zc_loss
                # backpropagate the gradients
                ge_loss.backward(retain_graph=True)
                # computes the new weights
                optimizer_GE.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Measure discriminator's ability to classify real from generated samples
            if wass_metric:
                # Gradient penaltytorch.autograd.set_detect_anomaly(True) term
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs,cuda=CUDA)

                # Wasserstein GAN loss w/gradient penalty
                #d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty # original
                d_loss = - torch.mean(D_real) + torch.mean(D_gen) + grad_penalty # corrected
                
            else:
                # Vanilla GAN loss
                real_loss = bce_loss(D_real, valid)
                fake_loss = bce_loss(D_gen, fake)
                d_loss = (real_loss + fake_loss) / 2

            d_loss.backward(inputs=list(discriminator.parameters()))
            optimizer_D.step()


        # Save training losses
        d_l.append(d_loss.item())
        ge_l.append(ge_loss.item())


        # Generator in eval mode
        generator.eval()
        encoder.eval()

        # Set number of examples for cycle calcs
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp
        
        if args.dataset == 'mnist':
            test_imgs, test_labels = next(iter(testdata))
            test_imgs = Variable(test_imgs.type(TENSOR))
        elif args.dataset == 'euromds':        
            test_imgs, test_labels, test_ids, test_outcomes = next(iter(testloader))
            times = test_outcomes[:, 0]
            events = test_outcomes[:, 1]
            test_imgs = Variable(test_imgs.type(TENSOR))

        ## Cycle through test real -> enc -> gen
        t_imgs, t_label = test_imgs.data, test_labels
        # Encode sample real instances
        e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)
        
        computed_labels = []
        for pred in e_tzc.detach().cpu().numpy():
            computed_labels.append(pred.argmax())
        computed_labels = np.array(computed_labels)
        
        # computing metrics
        acc = my_metrics.acc(t_label.detach().cpu().numpy(),
         computed_labels)
        nmi = my_metrics.nmi(t_label.detach().cpu().numpy(),
         computed_labels)
        ami = my_metrics.ami(t_label.detach().cpu().numpy(),
         computed_labels)
        ari = my_metrics.ari(t_label.detach().cpu().numpy(),
         computed_labels)
        ran = my_metrics.ran(t_label.detach().cpu().numpy(),
         computed_labels)
        homo = my_metrics.homo(t_label.detach().cpu().numpy(),
         computed_labels)
        print('FedIter %d\n\tacc %.5f\n\tnmi %.5f\n\tami %.5f\n\tari %.5f\n\tran %.5f\n\thomo %.5f' % \
            (epoch, acc, nmi, ami, ari, ran, homo))
        if args.dataset == 'euromds':
            # plotting outcomes on the labels
            my_fn.plot_lifelines_pred(times,
                                      events,
                                      computed_labels,
                                      path_to_out)
        if epoch % 10 == 0:  # print confusion matrix
            my_fn.print_confusion_matrix(
                t_label.detach().cpu().numpy(),
                computed_labels,
                path_to_out)
        # dumping and retrieving the results
        metrics = {"accuracy": acc,
                    "normalized_mutual_info_score": nmi,
                    "adjusted_mutual_info_score": ami,
                    "adjusted_rand_score": ari,
                    "rand_score": ran,
                    "homogeneity_score": homo}
        result = metrics.copy()        
        
        # Generate sample instances from encoding
        teg_imgs = generator(e_tzn, e_tzc)
        # Calculate cycle reconstruction loss
        img_mse_loss = mse_loss(t_imgs, teg_imgs)
        # Save img reco cycle loss
        c_i.append(img_mse_loss.item())
    

        ## Cycle through randomly sampled encoding -> generator -> encoder
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp,
                                                latent_dim=latent_dim,
                                                n_c=n_c,
                                    cuda=CUDA)
        # Generate sample instances
        gen_imgs_samp = generator(zn_samp, zc_samp)

        # Encode sample instances
        zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp)

        # Calculate cycle latent losses
        lat_mse_loss = mse_loss(zn_e, zn_samp)
        lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)

        # Save latent space cycle losses
        c_zn.append(lat_mse_loss.item())
        c_zc.append(lat_xe_loss.item())
    
        # Save cycled and generated examples!
        if args.save_img:
            r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
            e_zn, e_zc, e_zc_logits = encoder(r_imgs)
            reg_imgs = generator(e_zn, e_zc)
            save_image(reg_imgs.data[:n_samp],
                    path_to_out+('cycle_reg_%06i.png' %(epoch)), 
                    nrow=n_sqrt_samp, normalize=True)
            save_image(gen_imgs_samp.data[:n_samp],
                    path_to_out+('gen_%06i.png' %(epoch)), 
                    nrow=n_sqrt_samp, normalize=True)
            
            ## Generate samples for specified classes
            stack_imgs = []
            for idx in range(n_c):
                # Sample specific class
                zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_c,
                                                        latent_dim=latent_dim,
                                                        n_c=n_c,
                                                        fix_class=idx,
                                        cuda=CUDA)

                # Generate sample instances
                gen_imgs_samp = generator(zn_samp, zc_samp)

                if (len(stack_imgs) == 0):
                    stack_imgs = gen_imgs_samp
                else:
                    stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)

            # Save class-specified generated examples!
            save_image(stack_imgs,
                    path_to_out/('gen_classes_%06i.png' %(epoch)),
                    nrow=n_c, normalize=True)
    

        print ("[Epoch %d/%d] \n"\
            "\tModel Losses: [D: %f] [GE: %f]" % (epoch, 
                                                    n_epochs, 
                                                    d_loss.item(),
                                                    ge_loss.item())
            )
        
        print("\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]"%(img_mse_loss.item(), 
                                                            lat_mse_loss.item(), 
                                                            lat_xe_loss.item())
            )
        
        result['img_mse_loss'] = img_mse_loss.item()
        result['lat_mse_loss'] = lat_mse_loss.item()
        result['lat_xe_loss'] = lat_xe_loss.item()
        result['round'] = epoch
        my_fn.dump_result_dict('clustergan', result, path_to_out)
        pred = {'ID': test_ids,
                'label': computed_labels}
        my_fn.dump_pred_dict('pred', pred, path_to_out)
