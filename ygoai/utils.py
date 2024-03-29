from pathlib import Path


def load_deck(fn):
    with open(fn) as f:
        lines = f.readlines()
        deck = [int(line) for line in  lines if line[:-1].isdigit()]
        return deck


def get_root_directory():
	cur = Path(__file__).resolve()
	return str(cur.parent.parent)


def extract_deck_name(path):
	return Path(path).stem

_languages = {
    "english": "en",
    "chinese": "zh",
}

def init_ygopro(env_id, lang, deck, code_list_file, preload_tokens=False):
	short = _languages[lang]
	db_path = Path(get_root_directory(), 'assets', 'locale', short, 'cards.cdb')
	deck_fp = Path(deck)
	if deck_fp.is_dir():
		decks = {f.stem: str(f) for f in deck_fp.glob("*.ydk")}
		deck_dir = deck_fp
		deck_name = 'random'
	else:
		deck_name = deck_fp.stem
		decks = {deck_name: deck}
		deck_dir = deck_fp.parent
	if preload_tokens:
		token_deck = deck_dir / "_tokens.ydk"
		if not token_deck.exists():
			raise FileNotFoundError(f"Token deck not found: {token_deck}")
		decks["_tokens"] = str(token_deck)
	if 'YGOPro' in env_id:
		from ygoenv.ygopro import init_module
	elif 'EDOPro' in env_id:
		from ygoenv.edopro import init_module
	init_module(str(db_path), code_list_file, decks)
	return deck_name