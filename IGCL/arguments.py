import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--lr', dest='lr', type=float,default=0.0005,
            help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=1,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=360,
            help='')
    parser.add_argument('--value1', type=float, default=0.4, help='The first specific value.')
    parser.add_argument('--value2', type=float, default=0.6, help='The second specific value.')
    parser.add_argument('--value3', type=float, default=0.8, help='The second specific value.')
    parser.add_argument('--value4', type=float, default=0.99, help='The second specific value.')
    return parser.parse_args()

