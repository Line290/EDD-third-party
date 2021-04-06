from utils import create_input_files
import argparse


if __name__ == '__main__':
    # Create input files (along with word map)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_folder", type=str, default="pubtabnet",
                        help="Source table images' folder.")
    parser.add_argument('--output_folder', type=str, default='output_w_none_399k_memory_effi',
                        help='Output folder to save processed data')
    # Training
    parser.add_argument("--max_len_token_structure", type=int, default=300,
                        help="Maximal length of structure's token")
    parser.add_argument("--max_len_token_cell", type=int, default=100,
                        help="Maximal length of each cell's token.")
    parser.add_argument("--image_size", type=int, default=80000,
                        help="Maximal image's height and width.")
    args = parser.parse_args()

    create_input_files(image_folder=args.image_folder,
                       output_folder=args.output_folder,
                       max_len_token_structure=args.max_len_token_structure,
                       max_len_token_cell=args.max_len_token_cell,
                       image_size=args.image_size)
