import os
from glob import glob

from dataclasses import dataclass
import tyro

from ygoai.embed import read_cards

@dataclass
class Args:
    output: str = "code_list.txt"
    """the file containing the list of card codes"""
    cdb: str = "../assets/locale/en/cards.cdb"
    """the cards database file"""
    script_dir: str = "script"
    """path to the scripts directory"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    cards = read_cards(args.cdb)[1]

    pattern = os.path.join(args.script_dir, "c*.lua")
    # list all c*.lua files
    script_files = glob(pattern)

    codes = sorted([os.path.basename(f).split(".")[0][1:] for f in script_files])
    # exclude constant.lua
    codes_s = set([int(c) for c in codes[:-1]])
    codes_c = sorted([ c.code for c in cards ])

    difference = codes_s.difference(codes_c)
    if len(difference) > 0:
        raise ValueError("Missing in cards.cdb: {difference}")

    print(f"Total {len(codes_c)} cards, {len(codes_s)} scripts")

    lines = []
    for c in codes_c:
        line = f"{c} {1 if c in codes_s else 0}"
        lines.append(line)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))