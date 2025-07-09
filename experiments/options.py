import argparse

parser = argparse.ArgumentParser(description='DINOv2-LoRA')

parser.add_argument('--exp_name', type=str, default='dino_lora_prompt')

# --------------------
# DataLoader Options
# --------------------

parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)
parser.add_argument('--len', type=int, default=300000)
parser.add_argument('--gpu', type=int, default=1)

parser.add_argument('--fg', action='store_true', default=False)
parser.add_argument('--pidinet', action='store_true', default=False)

# ----------------------
# Training Params
# ----------------------

parser.add_argument('--encoder_lr', type=float, default=2e-4)
parser.add_argument('--encoder_LN_lr', type=float, default=1e-6)
parser.add_argument('--prompt_lr', type=float, default=1e-4)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=12)
parser.add_argument('--backbone', default='ViT-B/16', type=str)


# ----------------------
# LoRA Parameters
# ----------------------
parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='vision')
parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
parser.add_argument('--r', default=64, type=int, help='the rank of the low-rank matrices')
parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')

# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_learning', action='store_true', default=False)
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

# ----------------------
# SBIR Parameters
# ----------------------
parser.add_argument('--model', type=str, default="")
parser.add_argument('--output_file', type=str, default="")
parser.add_argument('--image_file', type=str, default="") 
parser.add_argument('--sketch_file', type=str, default="")

# ----------------------
parser.add_argument('-c', '--config', help='path to the config file', required=False)


opts = parser.parse_args()
