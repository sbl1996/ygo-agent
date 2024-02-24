import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import voyageai

import tyro

from ygoai.embed import read_cards
from ygoai.utils import load_deck

@dataclass
class Args:
    deck_dir: str = "../assets/deck"
    """the directory of ydk files"""
    code_list_file: str = "code_list.txt"
    """the file containing the list of card codes"""
    embeddings_file: Optional[str] = "embeddings.npy"
    """the npz file containing the embeddings of the cards"""
    cards_db: str = "../assets/locale/en/cards.cdb"
    """the cards database file"""
    batch_size: int = 64
    """the batch size for embedding generation"""
    wait_time: float = 0.1
    """the time to wait between each batch"""


def get_embeddings(texts, batch_size=64, wait_time=0.1, verbose=False):
    vo = voyageai.Client()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        if verbose:
            print(f"Embedding {i} / {len(texts)}")
        embeddings += vo.embed(
            texts[i : i + batch_size], model="voyage-2", truncation=False).embeddings
        time.sleep(wait_time)
    embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings


def read_decks(d):
    # iterate over ydk files
    codes = []
    for file in os.listdir(d):
        if file.endswith(".ydk"):
            file = os.path.join(d, file)
            codes += load_deck(file)
    return set(codes)


def read_texts(cards_db, codes):
    df, cards = read_cards(cards_db)
    code2card = {c.code: c for c in cards}
    texts = []
    for code in codes:
        texts.append(code2card[code].format())
    return texts


if __name__ == "__main__":
    args = tyro.cli(Args)

    deck_dir = args.deck_dir
    code_list_file = args.code_list_file
    embeddings_file = args.embeddings_file
    cards_db = args.cards_db

    # read code_list file
    if not os.path.exists(code_list_file):
        with open(code_list_file, "w") as f:
            f.write("")
    with open(code_list_file, "r") as f:
        code_list = f.readlines()
    code_list = [int(code.strip()) for code in code_list]
    print(f"The database contains {len(code_list)} cards.")

    if embeddings_file is not None:
        # read embeddings
        if not os.path.exists(embeddings_file):
            sample_embedding = get_embeddings(["test"])[0]
            all_embeddings = np.zeros((0, len(sample_embedding)), dtype=np.float32)
        else:
            all_embeddings = np.load(embeddings_file)
        print("Embedding dim:", all_embeddings.shape[1])

        assert len(all_embeddings) == len(code_list), f"The number of embeddings({len(all_embeddings)}) does not match the number of cards."

    all_codes = set(code_list)

    new_codes = []
    for code in read_decks(deck_dir):
        if code not in all_codes:
            new_codes.append(code)
    
    if new_codes == []:
        print("No new cards have been added to the database.")
        exit()

    new_texts = read_texts(cards_db, new_codes)
    print(new_texts)
    if embeddings_file is not None:
        embeddings = get_embeddings(new_texts, args.batch_size, args.wait_time, verbose=True)
        all_embeddings = np.concatenate([all_embeddings, np.array(embeddings)], axis=0)
        np.save(embeddings_file, all_embeddings)

    # update code_list
    code_list += new_codes

    with open(code_list_file, "w") as f:
        f.write("\n".join(map(str, code_list)) + "\n")

    print(f"{len(new_codes)} new cards have been added to the database.")