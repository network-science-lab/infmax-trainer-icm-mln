import argparse


def parse_args(fixed_args = None):

    parser = argparse.ArgumentParser(description="Run multi-node2vec on multilayer networks.")

    parser.add_argument(
        '--dir', nargs='?', default='data/CONTROL_fmt',
        help='Absolute path to directory of correlation/adjacency matrix files (csv format). Note that rows and columns must be properly labeled by node ID in each .csv.')

    parser.add_argument('--output', nargs='?', default='new_results/',
                        help='Absolute path to output directory (no extension).')

    #parser.add_argument('--filename', nargs='?', default='new_results/mltn2v_control',
    #                    help='output filename (no extension).')

    parser.add_argument('--d', type=int, default=100,
                        help='Dimensionality. Default is 100.')

    parser.add_argument('--walk_length', type=int, default=100,
                        help='Length of each random walk. Default is 100.')
                        
    parser.add_argument('--window_size', type=int, default = 10,
                        help='Size of context window used for Skip Gram optimization. Default is 10.')

    parser.add_argument('--n_samples', type=int, default=1,
                        help='Number of walks per node per layer. Default is 1.')

    parser.add_argument('--thresh', type=float,
                        help='Threshold for converting a weighted network to an unweighted one. All weights less than or equal to thresh will be considered 0 and all others 1. Default is 0.5. Use None if the network is unweighted.')

    # parser.add_argument('--w2v_iter', default=1, type=int,
    #                         help='Number of epochs in word2vec')

    parser.add_argument('--w2v_workers', type=int, default=8,
                        help='Number of parallel worker threads. Default is 8.')
                        
    parser.add_argument('--rvals', type=float, default=0.25,
                        help='Layer walk parameter for neighborhood search. Default is 0.25')

    parser.add_argument('--pvals', type=float, default=1,
                        help='Return walk parameter for neighborhood search. Default is 1')
    
    parser.add_argument('--qvals', type=float, default=0.5,
                        help='Exploration walk parameter for neighborhood search. Default is 0.50')

    args = parser.parse_args(fixed_args)
    return args
