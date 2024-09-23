"""
Wrapper for the multi-node2vec algorithm. 

Details can be found in the paper: "Fast Embedding of Multilayer Networks: An Algorithm and Application to Group fMRI" 
by JD Wilson, M Baybay, R Sankar, and P Stillman

Preprint here: https://arxiv.org/pdf/1809.06437.pdf

Contributors:
- Melanie Baybay
University of San Francisco, Department of Computer Science
- Rishi Sankar
Henry M. Gunn High School
- James D. Wilson (maintainer)
University of San Francisco, Department of Mathematics and Statistics

Questions or Bugs? Contact James D. Wilson at jdwilson4@usfca.edu
"""
import logging
import os
import time

import src as mltn2v
from src.cli_args import parse_args


def main(args):
    start = time.time()

    # PARSE LAYERS -- THRESHOLD & CONVERT TO BINARY
    layers = mltn2v.timed_invoke(
        "parsing network layers",
        lambda: mltn2v.parse_matrix_layers(args.dir, binary=True, thresh=args.thresh),
    )

    # check if layers were parsed
    if not layers:
        logging.warning("Whoops!")
        return

    # EXTRACT NEIGHBORHOODS
    nbrhd_dict = mltn2v.timed_invoke(
        "extracting neighborhoods",
        lambda: mltn2v.extract_neighborhoods_walk(
            layers, args.walk_length, [args.rvals], args.pvals, args.qvals
        ),
    )

    # GENERATE FEATURES
    out = mltn2v.clean_output(args.output)
    out_path = os.path.join(out, "mltn2v_results")
    mltn2v.timed_invoke(
        "generating features",
        lambda: mltn2v.generate_features(
            nbrhd_dict[args.rvals],
            args.d,
            out_path,
            nbrhd_size=args.window_size,
            w2v_iter=1,
            workers=args.w2v_workers,
        ),
    )
    logging.info(
        "Completed Multilayer Network Embedding for r="
        + str(args.rvals)
        + " in {:.2f} secs.\nSee results:".format(time.time() - start)
    )
    logging.info("\t" + out_path + ".csv")


if __name__ == "__main__":
    args = parse_args(
        # [
        #     "--dir", "data/toy_network",
        #     "--output", "results/toy_network",
        #     "--d", "2",
        #     "--window_size", "2",
        #     "--n_samples", "1",
        #     "--thresh", "0.5",
        #     "--rvals", "0.25"
        # ]
    )
    # logging.info(args)
    main(args)
