class cargs():
    def __init__(self):
        self.batch_size = 128
        self.log = './log/cifar10'
        self.resume = False
        self.multi_gpu = False
        self.epoch = 20000
        self.z_dim = 128
        self.alpha = 1.0
        self.beta = 1.0
        self.lr = 0.0002
        self.disc_iter = 5

    def __str__(self):
        out = "=========ARGUMENTS==========\n"
        for k, v in vars(self).items():
            out += '{}: {}\n'.format(k, v)
        out += "==============================\n"
        out += "==============================\n"
        return out

args = cargs()

