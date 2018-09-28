import  warnings

class DefaultConfig(object):

    env             = 'default'

    # model           = 'SeResnet50'
    # model = 'se_resnet101'
    model = 'senet154'
    # model           = 'ResNet34'
    # model   = 'AlexNet'

    train_data_root = '/home/ailab/yaotc/defects/data/train/'
    test_data_root  = '/home/ailab/yaotc/defects/data/test/'
    load_model_path = './checkpoints/model.pth'
    save_path       = './checkpoints/'


    batch_size   = 8    # 128
    use_gpu      = True # True
    sampler_num1 = 12000 # 1800
    sampler_num2 = 1200  # 500
    num_workers  = 4
    print_freq   = 20
    save_epoch_freq = 20 

    debug_file   = '/tmp/debug'
    result_file  = 'result.csv'

    max_epoch    = 80
    lr           = 0.0001
    lr_decay     = 0.95

    weight_decay = 1e-4


def parse(self , kwargs):

    for k , v in kwargs.iteritems():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribute %s" %k)

        setattr(self , k , v)

    print('user config:')

    for k, v in self.__class__.__dict__.iteritems():
        if not k.startswith('__'):
            print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()


