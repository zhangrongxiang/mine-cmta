import argparse


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Configurations for Survival Analysis on TCGA Data.")
    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument(
        "--data_root_dir", type=str, default="path/to/data_root_dir", help="Data directory to WSI features (extracted via CLAM"
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducible experiment (default: 1)")

    parser.add_argument(
        "--which_splits", type=str, default="5foldcv", help="Which splits folder to use in ./splits/ (Default: ./splits/5foldcv"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tcga_blca_100",
        help='Which cancer type within ./splits/<which_dataset> to use for training. Used synonymously for "task" (Default: tcga_blca_100)',
    )
    parser.add_argument("--log_data", action="store_true", default=True, help="Log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="Evaluate model on test set")
    parser.add_argument("--resume", type=str, default="", metavar="PATH", help="Path to latest checkpoint (default: none)")

    parser.add_argument("--OOM", type=int, default=0, help="Ramdomly sampling some patches to avoid OOM error")

    # Model Parameters.
    parser.add_argument(
        "--model",
        type=str,
        default="mcat",
        help="Type of model (Default: mcat)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=[
            "small",
            "large",
        ],
        default="small",
        help="Size of some models (Transformer)",
    )
    parser.add_argument(
        "--modal",
        type=str,
        choices=["omic", "path", "pathomic", "cluster", "coattn"],
        default="coattn",
        help="Specifies which modalities to use / collate function in dataloader.",
    )

    parser.add_argument("--alpha", type=float, default=0.5, help="hyper-parameter of loss function")
    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam",
                        "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine"], default="cosine")

    parser.add_argument("--num_epoch", type=int, default=20, help="Maximum number of epochs to train (default: 20)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size (Default: 1, due to varying bag sizes)")


    parser.add_argument(
        "--loss",
        type=str,
        default="nll_surv",
        help="slide-level classification loss function (default: ce)",
    )

    parser.add_argument("--weighted_sample", action="store_true", default=True, help="Enable weighted sampling")


    # =======================================
    parser.add_argument("--F_alpha", type=float, default=0.5, help="The proportion of genes involved in fusion")
    parser.add_argument("--F_beta", type=float, default=0.5,
                        help="The proportion of fine particle size in multi-particle size fusion")

    parser.add_argument("--GT", type=float, default=0.5, help="Temperature of gene token selection means current num*T")
    parser.add_argument("--PT", type=float, default=0.5,
                        help="Temperature of img token selection means current num*T")

    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--tokenS", type=str, choices=["both", "G", "P","N"], default="both")
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["concat", "bilinear","hyperbolic","fineCoarse","Mamba","Aconcat"],
        default="concat",
        help="Modality fuison strategy",
    )
    parser.add_argument("--MoELoss", action="store_false", default=False, help=" MoE Negative distance loss")
    parser.add_argument("--LossRate", type=float, default=1e-5, help="MoELoss rate")
    parser.add_argument("--HRate", type=float, default=1e-10, help="Hyperbolic rate")
    parser.add_argument("--modality", type=str, choices=["Both", "G", "P"],default="Both",help= "Modality to test")
    # =======================================
    args = parser.parse_args()
    return args
