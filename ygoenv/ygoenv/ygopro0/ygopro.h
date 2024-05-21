#ifndef YGOENV_YGOPRO0_YGOPRO_H_
#define YGOENV_YGOPRO0_YGOPRO_H_

// clang-format off
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <numeric>
#include <stdexcept>
#include <string>
#include <cstring>
#include <fstream>
#include <shared_mutex>
#include <iostream>
#include <set>


#include <fmt/core.h>
#include <fmt/ranges.h>
#include <SQLiteCpp/SQLiteCpp.h>
#include <SQLiteCpp/VariadicBind.h>
#include <ankerl/unordered_dense.h>
#include <unordered_set>

#include "ygoenv/core/BS_thread_pool.h"

#include "ygoenv/core/async_envpool.h"
#include "ygoenv/core/env.h"

#include "ygopro-core/common.h"
#include "ygopro-core/card_data.h"
#include "ygopro-core/duel.h"
#include "ygopro-core/ocgapi.h"

// clang-format on

namespace ygopro0 {

inline std::vector<std::vector<int>> combinations(int n, int r) {
  std::vector<std::vector<int>> combs;
  std::vector<bool> m(n);
  std::fill(m.begin(), m.begin() + r, true);

  do {
    std::vector<int> cs;
    cs.reserve(r);
    for (int i = 0; i < n; ++i) {
      if (m[i]) {
        cs.push_back(i);
      }
    }
    combs.push_back(cs);
  } while (std::prev_permutation(m.begin(), m.end()));

  return combs;
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind, int i,
                   int r) {
  if (r <= 0) {
    return false;
  }
  int n = ind.size();
  if (i == n - 1) {
    return r == 1 || (w[ind[i]] == r);
  }
  return sum_to(w, ind, i + 1, r - 1) || sum_to(w, ind, i + 1, r - w[ind[i]]);
}

inline bool sum_to(const std::vector<int> &w, const std::vector<int> ind,
                   int r) {
  return sum_to(w, ind, 0, r);
}

inline std::vector<std::vector<int>>
combinations_with_weight(const std::vector<int> &weights, int r) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to(weights, comb, r)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

inline bool sum_to2(const std::vector<std::vector<int>> &w,
                    const std::vector<int> ind, int i, int r) {
  if (r <= 0) {
    return false;
  }
  int n = ind.size();
  const auto &w_ = w[ind[i]];
  if (i == n - 1) {
    if (w_.size() == 1) {
      return w_[0] == r;
    } else {
      return w_[0] == r || w_[1] == r;
    }
  }
  if (w_.size() == 1) {
    return sum_to2(w, ind, i + 1, r - w_[0]);
  } else {
    return sum_to2(w, ind, i + 1, r - w_[0]) ||
           sum_to2(w, ind, i + 1, r - w_[1]);
  }
}

inline bool sum_to2(const std::vector<std::vector<int>> &w,
                    const std::vector<int> ind, int r) {
  return sum_to2(w, ind, 0, r);
}

inline std::vector<std::vector<int>>
combinations_with_weight2(
  const std::vector<std::vector<int>> &weights, int r) {
  int n = weights.size();
  std::vector<std::vector<int>> results;

  for (int k = 1; k <= n; k++) {
    std::vector<std::vector<int>> combs = combinations(n, k);
    for (const auto &comb : combs) {
      if (sum_to2(weights, comb, r)) {
        results.push_back(comb);
      }
    }
  }
  return results;
}

static std::string msg_to_string(int msg) {
  switch (msg) {
  case MSG_RETRY:
    return "retry";
  case MSG_HINT:
    return "hint";
  case MSG_WIN:
    return "win";
  case MSG_SELECT_BATTLECMD:
    return "select_battlecmd";
  case MSG_SELECT_IDLECMD:
    return "select_idlecmd";
  case MSG_SELECT_EFFECTYN:
    return "select_effectyn";
  case MSG_SELECT_YESNO:
    return "select_yesno";
  case MSG_SELECT_OPTION:
    return "select_option";
  case MSG_SELECT_CARD:
    return "select_card";
  case MSG_SELECT_CHAIN:
    return "select_chain";
  case MSG_SELECT_PLACE:
    return "select_place";
  case MSG_SELECT_POSITION:
    return "select_position";
  case MSG_SELECT_TRIBUTE:
    return "select_tribute";
  case MSG_SELECT_COUNTER:
    return "select_counter";
  case MSG_SELECT_SUM:
    return "select_sum";
  case MSG_SELECT_DISFIELD:
    return "select_disfield";
  case MSG_SORT_CARD:
    return "sort_card";
  case MSG_SELECT_UNSELECT_CARD:
    return "select_unselect_card";
  case MSG_CONFIRM_DECKTOP:
    return "confirm_decktop";
  case MSG_CONFIRM_CARDS:
    return "confirm_cards";
  case MSG_SHUFFLE_DECK:
    return "shuffle_deck";
  case MSG_SHUFFLE_HAND:
    return "shuffle_hand";
  case MSG_SWAP_GRAVE_DECK:
    return "swap_grave_deck";
  case MSG_SHUFFLE_SET_CARD:
    return "shuffle_set_card";
  case MSG_REVERSE_DECK:
    return "reverse_deck";
  case MSG_DECK_TOP:
    return "deck_top";
  case MSG_SHUFFLE_EXTRA:
    return "shuffle_extra";
  case MSG_NEW_TURN:
    return "new_turn";
  case MSG_NEW_PHASE:
    return "new_phase";
  case MSG_CONFIRM_EXTRATOP:
    return "confirm_extratop";
  case MSG_MOVE:
    return "move";
  case MSG_POS_CHANGE:
    return "pos_change";
  case MSG_SET:
    return "set";
  case MSG_SWAP:
    return "swap";
  case MSG_FIELD_DISABLED:
    return "field_disabled";
  case MSG_SUMMONING:
    return "summoning";
  case MSG_SUMMONED:
    return "summoned";
  case MSG_SPSUMMONING:
    return "spsummoning";
  case MSG_SPSUMMONED:
    return "spsummoned";
  case MSG_FLIPSUMMONING:
    return "flipsummoning";
  case MSG_FLIPSUMMONED:
    return "flipsummoned";
  case MSG_CHAINING:
    return "chaining";
  case MSG_CHAINED:
    return "chained";
  case MSG_CHAIN_SOLVING:
    return "chain_solving";
  case MSG_CHAIN_SOLVED:
    return "chain_solved";
  case MSG_CHAIN_END:
    return "chain_end";
  case MSG_CHAIN_NEGATED:
    return "chain_negated";
  case MSG_CHAIN_DISABLED:
    return "chain_disabled";
  case MSG_RANDOM_SELECTED:
    return "random_selected";
  case MSG_BECOME_TARGET:
    return "become_target";
  case MSG_DRAW:
    return "draw";
  case MSG_DAMAGE:
    return "damage";
  case MSG_RECOVER:
    return "recover";
  case MSG_EQUIP:
    return "equip";
  case MSG_LPUPDATE:
    return "lpupdate";
  case MSG_CARD_TARGET:
    return "card_target";
  case MSG_CANCEL_TARGET:
    return "cancel_target";
  case MSG_PAY_LPCOST:
    return "pay_lpcost";
  case MSG_ADD_COUNTER:
    return "add_counter";
  case MSG_REMOVE_COUNTER:
    return "remove_counter";
  case MSG_ATTACK:
    return "attack";
  case MSG_BATTLE:
    return "battle";
  case MSG_ATTACK_DISABLED:
    return "attack_disabled";
  case MSG_DAMAGE_STEP_START:
    return "damage_step_start";
  case MSG_DAMAGE_STEP_END:
    return "damage_step_end";
  case MSG_MISSED_EFFECT:
    return "missed_effect";
  case MSG_TOSS_COIN:
    return "toss_coin";
  case MSG_TOSS_DICE:
    return "toss_dice";
  case MSG_ROCK_PAPER_SCISSORS:
    return "rock_paper_scissors";
  case MSG_HAND_RES:
    return "hand_res";
  case MSG_ANNOUNCE_RACE:
    return "announce_race";
  case MSG_ANNOUNCE_ATTRIB:
    return "announce_attrib";
  case MSG_ANNOUNCE_CARD:
    return "announce_card";
  case MSG_ANNOUNCE_NUMBER:
    return "announce_number";
  case MSG_CARD_HINT:
    return "card_hint";
  case MSG_TAG_SWAP:
    return "tag_swap";
  case MSG_RELOAD_FIELD:
    return "reload_field";
  case MSG_AI_NAME:
    return "ai_name";
  case MSG_SHOW_HINT:
    return "show_hint";
  case MSG_PLAYER_HINT:
    return "player_hint";
  case MSG_MATCH_KILL:
    return "match_kill";
  case MSG_CUSTOM_MSG:
    return "custom_msg";
  default:
    return "unknown_msg";
  }
}

// system string
static const ankerl::unordered_dense::map<int, std::string> system_strings = {
    {30, "Replay rules apply. Continue this attack?"},
    {31, "Attack directly with this monster?"},
    {96, "Use the effect of [%ls] to avoid destruction?"},
    {221, "On [%ls], Activate Trigger Effect of [%ls]?"},
    {1190, "Add to hand"},
    {1192, "Banish"},
    {1621, "Attack Negated"},
    {1622, "[%ls] Missed timing"}
};

static std::string get_system_string(int desc) {
  auto it = system_strings.find(desc);
  if (it != system_strings.end()) {
    return it->second;
  }
  return "system string " + std::to_string(desc);
}

static std::string ltrim(std::string s) {
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(),
                       std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

inline std::vector<std::string> flag_to_usable_cardspecs(uint32_t flag,
                                                         bool reverse = false) {
  std::string zone_names[4] = {"m", "s", "om", "os"};
  std::vector<std::string> specs;
  for (int j = 0; j < 4; j++) {
    uint32_t value = (flag >> (j * 8)) & 0xff;
    for (int i = 0; i < 8; i++) {
      bool avail = (value & (1 << i)) == 0;
      if (reverse) {
        avail = !avail;
      }
      if (avail) {
        specs.push_back(zone_names[j] + std::to_string(i + 1));
      }
    }
  }
  return specs;
}

inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos) {
  std::string spec;
  if (loc & LOCATION_HAND) {
    spec += "h";
  } else if (loc & LOCATION_MZONE) {
    spec += "m";
  } else if (loc & LOCATION_SZONE) {
    spec += "s";
  } else if (loc & LOCATION_GRAVE) {
    spec += "g";
  } else if (loc & LOCATION_REMOVED) {
    spec += "r";
  } else if (loc & LOCATION_EXTRA) {
    spec += "x";
  }
  spec += std::to_string(seq + 1);
  if (loc & LOCATION_OVERLAY) {
    spec.push_back('a' + pos);
  }
  return spec;
}

inline std::string ls_to_spec(uint8_t loc, uint8_t seq, uint8_t pos, bool opponent) {
  std::string spec = ls_to_spec(loc, seq, pos);
  if (opponent) {
    spec.insert(0, 1, 'o');
  }
  return spec;
}

inline std::tuple<uint8_t, uint8_t, uint8_t>
spec_to_ls(const std::string spec) {
  uint8_t loc;
  uint8_t seq;
  uint8_t pos = 0;
  int offset = 1;
  if (spec[0] == 'h') {
    loc = LOCATION_HAND;
  } else if (spec[0] == 'm') {
    loc = LOCATION_MZONE;
  } else if (spec[0] == 's') {
    loc = LOCATION_SZONE;
  } else if (spec[0] == 'g') {
    loc = LOCATION_GRAVE;
  } else if (spec[0] == 'r') {
    loc = LOCATION_REMOVED;
  } else if (spec[0] == 'x') {
    loc = LOCATION_EXTRA;
  } else if (std::isdigit(spec[0])) {
    loc = LOCATION_DECK;
    offset = 0;
  } else {
    throw std::runtime_error("Invalid location");
  }
  int end = offset;
  while (end < spec.size() && std::isdigit(spec[end])) {
    end++;
  }
  seq = std::stoi(spec.substr(offset, end - offset)) - 1;
  if (end < spec.size()) {
    pos = spec[end] - 'a';
  }
  return {loc, seq, pos};
}

inline uint32_t ls_to_spec_code(uint8_t loc, uint8_t seq, uint8_t pos,
                                bool opponent) {
  uint32_t c = opponent ? 1 : 0;
  c |= (loc << 8);
  c |= (seq << 16);
  c |= (pos << 24);
  return c;
}

inline uint32_t spec_to_code(const std::string &spec) {
  int offset = 0;
  bool opponent = false;
  if (spec[0] == 'o') {
    opponent = true;
    offset++;
  }
  auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
  return ls_to_spec_code(loc, seq, pos, opponent);
}

inline std::string code_to_spec(uint32_t spec_code) {
  uint8_t loc = (spec_code >> 8) & 0xff;
  uint8_t seq = (spec_code >> 16) & 0xff;
  uint8_t pos = (spec_code >> 24) & 0xff;
  bool opponent = (spec_code & 0xff) == 1;
  return ls_to_spec(loc, seq, pos, opponent);
}

static std::tuple<std::vector<uint32>, std::vector<uint32>, std::vector<uint32>> read_decks(const std::string &fp) {
  std::ifstream file(fp);
  std::string line;
  std::vector<uint32> main_deck, extra_deck, side_deck;
  bool found_extra = false;

  if (file.is_open()) {
    // Read the main deck
    while (std::getline(file, line)) {
      if (line.find("side") != std::string::npos) {
        break;
      }
      if (line.find("extra") != std::string::npos) {
        found_extra = true;
        break;
      }
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        main_deck.push_back(std::stoul(line));
      }
    }

    if (main_deck.size() < 40) {
      std::string err = fmt::format("Main deck must contain at least 40 cards, found: {}, file: {}", main_deck.size(), fp);
      throw std::runtime_error(err);
    }

    // Read the extra deck
    if (found_extra) {
      while (std::getline(file, line)) {
        if (line.find("side") != std::string::npos) {
          break;
        }
        // Check if line contains only digits
        if (std::all_of(line.begin(), line.end(), ::isdigit)) {
          extra_deck.push_back(std::stoul(line));
        }
      }
    }

    // Read the side deck
    while (std::getline(file, line)) {
      // Check if line contains only digits
      if (std::all_of(line.begin(), line.end(), ::isdigit)) {
        side_deck.push_back(std::stoul(line));
      }
    }

    file.close();
  } else {
    throw std::runtime_error(fmt::format("Unable to open deck file: {}", fp));
  }

  return std::make_tuple(main_deck, extra_deck, side_deck);
}

template <class K = uint8_t>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::map<K, std::string> &m, int id_offset = 0,
         int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  int i = 0;
  for (const auto &[k, v] : m) {
    if (i < m_offset) {
      i++;
      continue;
    }
    m2[k] = i - m_offset + id_offset;
    i++;
  }
  return m2;
}

template <class K = char>
ankerl::unordered_dense::map<K, uint8_t>
make_ids(const std::vector<K> &cmds, int id_offset = 0, int m_offset = 0) {
  ankerl::unordered_dense::map<K, uint8_t> m2;
  for (int i = m_offset; i < cmds.size(); i++) {
    m2[cmds[i]] = i - m_offset + id_offset;
  }
  return m2;
}

static std::string reason_to_string(uint8_t reason) {
  // !victory 0x0 Surrendered
  // !victory 0x1 LP reached 0
  // !victory 0x2 Cards can't be drawn
  // !victory 0x3 Time limit up
  // !victory 0x4 Lost connection
  switch (reason) {
  case 0x0:
    return "Surrendered";
  case 0x1:
    return "LP reached 0";
  case 0x2:
    return "Cards can't be drawn";
  case 0x3:
    return "Time limit up";
  case 0x4:
    return "Lost connection";
  default:
    return "Unknown";
  }
}

#define DEFINE_X_TO_ID_FUN(name, x_map) \
inline uint8_t name(decltype(x_map)::key_type x) { \
  auto it = x_map.find(x); \
  if (it != x_map.end()) { \
    return it->second; \
  } \
  throw std::runtime_error( \
    fmt::format("[" #name "] cannot find id: {}", x)); \
}

#define DEFINE_X_TO_STRING_FUN(name, x_map) \
inline std::string name(decltype(x_map)::key_type x) { \
  auto it = x_map.find(x); \
  if (it != x_map.end()) { \
    return it->second; \
  } \
  return "unknown"; \
}

static const std::map<uint8_t, std::string> location2str = {
    {LOCATION_DECK, "Deck"},
    {LOCATION_HAND, "Hand"},
    {LOCATION_MZONE, "Main Monster Zone"},
    {LOCATION_SZONE, "Spell & Trap Zone"},
    {LOCATION_GRAVE, "Graveyard"},
    {LOCATION_REMOVED, "Banished"},
    {LOCATION_EXTRA, "Extra Deck"},
};

static const ankerl::unordered_dense::map<uint8_t, uint8_t> location2id =
    make_ids(location2str, 1);
DEFINE_X_TO_ID_FUN(location_to_id, location2id)


#define POS_NONE 0x0 // xyz materials (overlay) ???

static const std::map<uint8_t, std::string> position2str = {
    {POS_NONE, "none"},
    {POS_FACEUP_ATTACK, "face-up attack"},
    {POS_FACEDOWN_ATTACK, "face-down attack"},
    {POS_ATTACK, "attack"},
    {POS_FACEUP_DEFENSE, "face-up defense"},
    {POS_FACEUP, "face-up"},
    {POS_FACEDOWN_DEFENSE, "face-down defense"},
    {POS_FACEDOWN, "face-down"},
    {POS_DEFENSE, "defense"},
};
DEFINE_X_TO_STRING_FUN(position_to_string, position2str)

static const ankerl::unordered_dense::map<uint8_t, uint8_t> position2id =
    make_ids(position2str);
DEFINE_X_TO_ID_FUN(position_to_id, position2id)


#define ATTRIBUTE_NONE 0x0 // token

static const std::map<uint8_t, std::string> attribute2str = {
    {ATTRIBUTE_NONE, "None"},   {ATTRIBUTE_EARTH, "Earth"},
    {ATTRIBUTE_WATER, "Water"}, {ATTRIBUTE_FIRE, "Fire"},
    {ATTRIBUTE_WIND, "Wind"},   {ATTRIBUTE_LIGHT, "Light"},
    {ATTRIBUTE_DARK, "Dark"},   {ATTRIBUTE_DEVINE, "Divine"},
};
DEFINE_X_TO_STRING_FUN(attribute_to_string, attribute2str)

static const ankerl::unordered_dense::map<uint8_t, uint8_t> attribute2id =
    make_ids(attribute2str);
DEFINE_X_TO_ID_FUN(attribute_to_id, attribute2id)


#define RACE_NONE 0x0 // token

static const std::map<uint32_t, std::string> race2str = {
    {RACE_NONE, "None"},
    {RACE_WARRIOR, "Warrior"},
    {RACE_SPELLCASTER, "Spellcaster"},
    {RACE_FAIRY, "Fairy"},
    {RACE_FIEND, "Fiend"},
    {RACE_ZOMBIE, "Zombie"},
    {RACE_MACHINE, "Machine"},
    {RACE_AQUA, "Aqua"},
    {RACE_PYRO, "Pyro"},
    {RACE_ROCK, "Rock"},
    {RACE_WINDBEAST, "Windbeast"},
    {RACE_PLANT, "Plant"},
    {RACE_INSECT, "Insect"},
    {RACE_THUNDER, "Thunder"},
    {RACE_DRAGON, "Dragon"},
    {RACE_BEAST, "Beast"},
    {RACE_BEASTWARRIOR, "Beast Warrior"},
    {RACE_DINOSAUR, "Dinosaur"},
    {RACE_FISH, "Fish"},
    {RACE_SEASERPENT, "Sea Serpent"},
    {RACE_REPTILE, "Reptile"},
    {RACE_PSYCHO, "Psycho"},
    {RACE_DEVINE, "Divine"},
    {RACE_CREATORGOD, "Creator God"},
    {RACE_WYRM, "Wyrm"},
    {RACE_CYBERSE, "Cyberse"},
    {RACE_ILLUSION, "Illusion"}};

static const ankerl::unordered_dense::map<uint32_t, uint8_t> race2id =
    make_ids(race2str);
DEFINE_X_TO_ID_FUN(race_to_id, race2id)


static const std::map<uint32_t, std::string> type2str = {
    {TYPE_MONSTER, "Monster"},
    {TYPE_SPELL, "Spell"},
    {TYPE_TRAP, "Trap"},
    {TYPE_NORMAL, "Normal"},
    {TYPE_EFFECT, "Effect"},
    {TYPE_FUSION, "Fusion"},
    {TYPE_RITUAL, "Ritual"},
    {TYPE_TRAPMONSTER, "Trap Monster"},
    {TYPE_SPIRIT, "Spirit"},
    {TYPE_UNION, "Union"},
    {TYPE_DUAL, "Dual"},
    {TYPE_TUNER, "Tuner"},
    {TYPE_SYNCHRO, "Synchro"},
    {TYPE_TOKEN, "Token"},
    {TYPE_QUICKPLAY, "Quick-play"},
    {TYPE_CONTINUOUS, "Continuous"},
    {TYPE_EQUIP, "Equip"},
    {TYPE_FIELD, "Field"},
    {TYPE_COUNTER, "Counter"},
    {TYPE_FLIP, "Flip"},
    {TYPE_TOON, "Toon"},
    {TYPE_XYZ, "XYZ"},
    {TYPE_PENDULUM, "Pendulum"},
    {TYPE_SPSUMMON, "Special"},
    {TYPE_LINK, "Link"},
};

inline std::vector<uint8_t> type_to_ids(uint32_t type) {
  std::vector<uint8_t> ids;
  ids.reserve(type2str.size());
  for (const auto &[k, v] : type2str) {
    ids.push_back(std::min(1u, type & k));
  }
  return ids;
}

static const std::map<int, std::string> phase2str = {
    {PHASE_DRAW, "draw phase"},
    {PHASE_STANDBY, "standby phase"},
    {PHASE_MAIN1, "main1 phase"},
    {PHASE_BATTLE_START, "battle start phase"},
    {PHASE_BATTLE_STEP, "battle step phase"},
    {PHASE_DAMAGE, "damage phase"},
    {PHASE_DAMAGE_CAL, "damage calculation phase"},
    {PHASE_BATTLE, "battle phase"},
    {PHASE_MAIN2, "main2 phase"},
    {PHASE_END, "end phase"},
};
DEFINE_X_TO_STRING_FUN(phase_to_string, phase2str)

static const ankerl::unordered_dense::map<int, uint8_t> phase2id =
    make_ids(phase2str);
DEFINE_X_TO_ID_FUN(phase_to_id, phase2id)


static const std::vector<int> _msgs = {
    MSG_SELECT_IDLECMD,  MSG_SELECT_CHAIN,     MSG_SELECT_CARD,
    MSG_SELECT_TRIBUTE,  MSG_SELECT_POSITION,  MSG_SELECT_EFFECTYN,
    MSG_SELECT_YESNO,    MSG_SELECT_BATTLECMD, MSG_SELECT_UNSELECT_CARD,
    MSG_SELECT_OPTION,   MSG_SELECT_PLACE,     MSG_SELECT_SUM,
    MSG_SELECT_DISFIELD, MSG_ANNOUNCE_ATTRIB,  MSG_ANNOUNCE_NUMBER,
};

static const ankerl::unordered_dense::map<int, uint8_t> msg2id =
    make_ids(_msgs, 1);
DEFINE_X_TO_ID_FUN(msg_to_id, msg2id)


static const ankerl::unordered_dense::map<char, uint8_t> cmd_act2id =
    make_ids({'t', 'r', 'c', 's', 'm', 'a', 'v'}, 1);
DEFINE_X_TO_ID_FUN(cmd_act_to_id, cmd_act2id)


static const ankerl::unordered_dense::map<char, uint8_t> cmd_phase2id =
    make_ids(std::vector<char>({'b', 'm', 'e'}), 1);
DEFINE_X_TO_ID_FUN(cmd_phase_to_id, cmd_phase2id)


static const ankerl::unordered_dense::map<char, uint8_t> cmd_yesno2id =
    make_ids(std::vector<char>({'y', 'n'}), 1);
DEFINE_X_TO_ID_FUN(cmd_yesno_to_id, cmd_yesno2id)


static const ankerl::unordered_dense::map<std::string, uint8_t> cmd_place2id =
    make_ids(std::vector<std::string>(
                 {"m1",  "m2",  "m3",  "m4",  "m5",  "m6",  "m7",  "s1",
                  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",  "s8",  "om1",
                  "om2", "om3", "om4", "om5", "om6", "om7", "os1", "os2",
                  "os3", "os4", "os5", "os6", "os7", "os8"}),
             1);
DEFINE_X_TO_ID_FUN(cmd_place_to_id, cmd_place2id)


inline std::pair<uint8_t, uint8_t> float_transform(int x) {
  x = x % 65536;
  return {
      static_cast<uint8_t>(x >> 8),
      static_cast<uint8_t>(x & 0xff),
  };
}

static std::vector<int> find_substrs(const std::string &str,
                                     const std::string &substr) {
  std::vector<int> res;
  int pos = 0;
  while ((pos = str.find(substr, pos)) != std::string::npos) {
    res.push_back(pos);
    pos += substr.length();
  }
  return res;
}

inline std::string time_now() {
  // strftime %Y-%m-%d %H-%M-%S
  time_t now = time(0);
  tm *ltm = localtime(&now);
  char buffer[80];
  strftime(buffer, 80, "%Y-%m-%d %H-%M-%S", ltm);
  return std::string(buffer);
}

// from ygopro/gframe/replay.h

// replay flag
#define REPLAY_COMPRESSED	0x1
#define REPLAY_TAG			0x2
#define REPLAY_DECODED		0x4
#define REPLAY_SINGLE_MODE	0x8
#define REPLAY_UNIFORM		0x10

// max size
#define MAX_REPLAY_SIZE	0x20000


struct ReplayHeader {
	unsigned int id;
	unsigned int version;
	unsigned int flag;
	unsigned int seed;
	unsigned int datasize;
	unsigned int start_time;
	unsigned char props[8];

	ReplayHeader()
		: id(0), version(0), flag(0), seed(0), datasize(0), start_time(0), props{ 0 } {}
};

// from ygopro/gframe/replay.h

using PlayerId = uint8_t;
using CardCode = uint32_t;
using CardId = uint16_t;

class Card {
  friend class YGOProEnv;

protected:
  CardCode code_ = 0;
  uint32_t alias_;
  uint64_t setcode_;
  uint32_t type_;
  uint32_t level_;
  uint32_t lscale_;
  uint32_t rscale_;
  int32_t attack_;
  int32_t defense_;
  uint32_t race_;
  uint32_t attribute_;
  uint32_t link_marker_;
  // uint32_t category_;
  std::string name_;
  std::string desc_;
  std::vector<std::string> strings_;

  uint32_t data_ = 0;

  uint32_t status_ = 0;
  PlayerId controler_ = 0;
  uint32_t location_ = 0;
  uint32_t sequence_ = 0;
  uint32_t position_ = 0;
  uint32_t counter_ = 0;

public:
  Card() = default;

  Card(CardCode code, uint32_t alias, uint64_t setcode, uint32_t type,
       uint32_t level, uint32_t lscale, uint32_t rscale, int32_t attack,
       int32_t defense, uint32_t race, uint32_t attribute, uint32_t link_marker,
       const std::string &name, const std::string &desc,
       const std::vector<std::string> &strings)
      : code_(code), alias_(alias), setcode_(setcode), type_(type),
        level_(level), lscale_(lscale), rscale_(rscale), attack_(attack),
        defense_(defense), race_(race), attribute_(attribute),
        link_marker_(link_marker), name_(name), desc_(desc), strings_(strings) {
  }

  ~Card() = default;

  void set_location(uint32_t location) {
    controler_ = location & 0xff;
    location_ = (location >> 8) & 0xff;
    sequence_ = (location >> 16) & 0xff;
    position_ = (location >> 24) & 0xff;
  }

  const std::string &name() const { return name_; }
  const std::string &desc() const { return desc_; }
  const uint32_t &type() const { return type_; }
  const uint32_t &level() const { return level_; }
  const std::vector<std::string> &strings() const { return strings_; }

  std::string get_spec(bool opponent) const {
    return ls_to_spec(location_, sequence_, position_, opponent);
  }

  std::string get_spec(PlayerId player) const {
    return get_spec(player != controler_);
  }

  uint32_t get_spec_code(PlayerId player) const {
    return ls_to_spec_code(location_, sequence_, position_,
                           player != controler_);
  }

  std::string get_position() const { return position_to_string(position_); }

  std::string get_effect_description(uint32_t desc,
                                     bool existing = false) const {
    std::string s;
    bool e = false;
    auto code = code_;
    if (desc > 10000) {
      code = desc >> 4;
    }
    uint32_t offset = desc - code_ * 16;
    bool in_range = (offset >= 0) && (offset < strings_.size());
    std::string str = "";
    if (in_range) {
      str = ltrim(strings_[offset]);
    }
    if (in_range || desc == 0) {
      if ((desc == 0) || str.empty()) {
        s = "Activate " + name_ + ".";
      } else {
        s = name_ + " (" + str + ")";
        e = true;
      }
    } else {
      s = get_system_string(desc);
      if (!s.empty()) {
        e = true;
      }
    }
    if (existing && !e) {
      s = "";
    }
    return s;
  }
};

struct MDuel {
  intptr_t pduel;
  uint64_t seed;
  std::vector<CardCode> main_deck0;
  std::vector<CardCode> extra_deck0;
  std::string deck_name0;
  std::vector<CardCode> main_deck1;
  std::vector<CardCode> extra_deck1;
  std::string deck_name1;
};

static std::mutex duel_mtx;

inline Card db_query_card(const SQLite::Database &db, CardCode code) {
  SQLite::Statement query1(db, "SELECT * FROM datas WHERE id=?");
  query1.bind(1, code);
  bool found = query1.executeStep();
  if (!found) {
    std::string msg = "[db_query_card] Card not found: " + std::to_string(code);
    throw std::runtime_error(msg);
  }

  uint32_t alias = query1.getColumn("alias");
  uint64_t setcode = query1.getColumn("setcode").getInt64();
  uint32_t type = query1.getColumn("type");
  uint32_t level_ = query1.getColumn("level");
  uint32_t level = level_ & 0xff;
  uint32_t lscale = (level_ >> 24) & 0xff;
  uint32_t rscale = (level_ >> 16) & 0xff;
  int32_t attack = query1.getColumn("atk");
  int32_t defense = query1.getColumn("def");
  uint32_t link_marker = 0;
  if (type & TYPE_LINK) {
    defense = 0;
    link_marker = defense;
  }
  uint32_t race = query1.getColumn("race");
  uint32_t attribute = query1.getColumn("attribute");

  SQLite::Statement query2(db, "SELECT * FROM texts WHERE id=?");
  query2.bind(1, code);
  query2.executeStep();

  std::string name = query2.getColumn(1);
  std::string desc = query2.getColumn(2);
  std::vector<std::string> strings;
  for (int i = 3; i < query2.getColumnCount(); ++i) {
    std::string str = query2.getColumn(i);
    strings.push_back(str);
  }
  return Card(code, alias, setcode, type, level, lscale, rscale, attack,
              defense, race, attribute, link_marker, name, desc, strings);
}

inline card_data db_query_card_data(const SQLite::Database &db, CardCode code) {
  SQLite::Statement query(db, "SELECT * FROM datas WHERE id=?");
  query.bind(1, code);
  query.executeStep();
  card_data card;
  card.code = code;
  card.alias = query.getColumn("alias");
  uint64_t setcode = query.getColumn("setcode").getInt64();
  card.set_setcode(setcode);
  card.type = query.getColumn("type");
  uint32_t level_ = query.getColumn("level");
  card.level = level_ & 0xff;
  card.lscale = (level_ >> 24) & 0xff;
  card.rscale = (level_ >> 16) & 0xff;
  card.attack = query.getColumn("atk");
  card.defense = query.getColumn("def");
  if (card.type & TYPE_LINK) {
    card.link_marker = card.defense;
    card.defense = 0;
  } else {
    card.link_marker = 0;
  }
  card.race = query.getColumn("race");
  card.attribute = query.getColumn("attribute");
  return card;
}

struct card_script {
  byte *buf;
  int len;
};

static ankerl::unordered_dense::map<CardCode, Card> cards_;
static ankerl::unordered_dense::map<CardCode, CardId> card_ids_;
static ankerl::unordered_dense::map<CardCode, card_data> cards_data_;
static ankerl::unordered_dense::map<std::string, card_script> cards_script_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    main_decks_;
static ankerl::unordered_dense::map<std::string, std::vector<CardCode>>
    extra_decks_;
static std::vector<std::string> deck_names_;

inline const Card &c_get_card(CardCode code) {
  auto it = cards_.find(code);
  if (it != cards_.end()) {
    return it->second;
  }
  throw std::runtime_error("[c_get_card] Card not found: " + std::to_string(code));
}

inline CardId &c_get_card_id(CardCode code) {
  auto it = card_ids_.find(code);
  if (it != card_ids_.end()) {
    return it->second;
  }
  throw std::runtime_error("[c_get_card_id] Card not found: " + std::to_string(code));
}

inline void sort_extra_deck(std::vector<CardCode> &deck) {
  std::vector<CardCode> c;
  std::vector<std::pair<CardCode, int>> fusion, xyz, synchro, link;

  for (auto code : deck) {
    const Card &cc = c_get_card(code);
    if (cc.type() & TYPE_FUSION) {
      fusion.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_XYZ) {
      xyz.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_SYNCHRO) {
      synchro.push_back({code, cc.level()});
    } else if (cc.type() & TYPE_LINK) {
      link.push_back({code, cc.level()});
    } else {
      throw std::runtime_error("Not extra deck card");
    }
  }

  auto cmp = [](const std::pair<CardCode, int> &a,
                const std::pair<CardCode, int> &b) {
    return a.second < b.second;
  };
  std::sort(fusion.begin(), fusion.end(), cmp);
  std::sort(xyz.begin(), xyz.end(), cmp);
  std::sort(synchro.begin(), synchro.end(), cmp);
  std::sort(link.begin(), link.end(), cmp);

  for (const auto &tc : fusion) {
    c.push_back(tc.first);
  }
  for (const auto &tc : xyz) {
    c.push_back(tc.first);
  }
  for (const auto &tc : synchro) {
    c.push_back(tc.first);
  }
  for (const auto &tc : link) {
    c.push_back(tc.first);
  }

  deck = c;
}

inline void preload_deck(const SQLite::Database &db,
                         const std::vector<CardCode> &deck) {
  for (const auto &code : deck) {
    auto it = cards_.find(code);
    if (it == cards_.end()) {
      cards_[code] = db_query_card(db, code);
      if (card_ids_.find(code) == card_ids_.end()) {
        throw std::runtime_error("Card not found in code list: " +
                                 std::to_string(code));
      }
    }

    auto it2 = cards_data_.find(code);
    if (it2 == cards_data_.end()) {
      cards_data_[code] = db_query_card_data(db, code);
    }
  }
}

inline uint32 card_reader_callback(CardCode code, card_data *card) {
  auto it = cards_data_.find(code);
  if (it == cards_data_.end()) {
    throw std::runtime_error("[card_reader_callback] Card not found: " + std::to_string(code));
  }
  *card = it->second;
  return 0;
}

static std::shared_timed_mutex scripts_mtx;

inline byte *read_card_script(const std::string &path, int *lenptr) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *lenptr = 0;
    return nullptr;
  }
  file.seekg(0, std::ios::end);
  int len = file.tellg();
  file.seekg(0, std::ios::beg);
  byte *buf = new byte[len];
  file.read((char *)buf, len);
  *lenptr = len;
  return buf;
}

inline byte *script_reader_callback(const char *name, int *lenptr) {
  std::string path(name);
  std::shared_lock<std::shared_timed_mutex> lock(scripts_mtx);
  auto it = cards_script_.find(path);
  if (it == cards_script_.end()) {
    lock.unlock();
    int len;
    byte *buf = read_card_script(path, &len);
    std::unique_lock<std::shared_timed_mutex> ulock(scripts_mtx);
    cards_script_[path] = {buf, len};
    it = cards_script_.find(path);
  }
  *lenptr = it->second.len;
  return it->second.buf;
}

static void init_module(const std::string &db_path,
                        const std::string &code_list_file,
                        const std::map<std::string, std::string> &decks) {
  // parse code from code_list_file
  std::ifstream file(code_list_file);
  std::string line;
  int i = 0;
  while (std::getline(file, line)) {
    i++;
    CardCode code = std::stoul(line);
    card_ids_[code] = i;
  }

  SQLite::Database db(db_path, SQLite::OPEN_READONLY);

  for (const auto &[name, deck] : decks) {
    auto [main_deck, extra_deck, side_deck] = read_decks(deck);
    main_decks_[name] = main_deck;
    extra_decks_[name] = extra_deck;
    if (name[0] != '_') {
      deck_names_.push_back(name);
    }

    preload_deck(db, main_deck);
    preload_deck(db, extra_deck);
    preload_deck(db, side_deck);
  }

  for (auto &[name, deck] : extra_decks_) {
    sort_extra_deck(deck);
  }

  set_card_reader(card_reader_callback);
  set_script_reader(script_reader_callback);
}

inline std::string getline() {
  char *line = nullptr;
  size_t len = 0;
  ssize_t read;

  read = getline(&line, &len, stdin);

  if (read != -1) {
    // Remove line ending character(s)
    if (line[read - 1] == '\n')
      line[read - 1] = '\0'; // Replace newline character with null terminator
    else if (line[read - 2] == '\r' && line[read - 1] == '\n') {
      line[read - 2] = '\0'; // Replace carriage return and newline characters
                             // with null terminator
      line[read - 1] = '\0';
    }

    std::string input(line);
    free(line);
    return input;
  } else {
    exit(0);
  }

  free(line);
  return "";
}

class Player {
  friend class YGOProEnv;

protected:
  const std::string nickname_;
  const int init_lp_;
  const PlayerId duel_player_;
  const bool verbose_;

  bool seen_waiting_ = false;

public:
  Player(const std::string &nickname, int init_lp, PlayerId duel_player,
         bool verbose = false)
      : nickname_(nickname), init_lp_(init_lp), duel_player_(duel_player),
        verbose_(verbose) {}
  virtual ~Player() = default;

  void notify(const std::string &text) {
    if (verbose_) {
      fmt::println("{} {}", duel_player_, text);
    }
  }

  const int &init_lp() const { return init_lp_; }

  virtual int think(const std::vector<std::string> &options) = 0;
};

class GreedyAI : public Player {
protected:
public:
  GreedyAI(const std::string &nickname, int init_lp, PlayerId duel_player,
           bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<std::string> &options) override { return 0; }
};

class RandomAI : public Player {
protected:
  std::mt19937 gen_;
  std::uniform_int_distribution<int> dist_;

public:
  RandomAI(int max_options, int seed, const std::string &nickname, int init_lp,
           PlayerId duel_player, bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose), gen_(seed),
        dist_(0, max_options - 1) {}

  int think(const std::vector<std::string> &options) override {
    return dist_(gen_) % options.size();
  }
};

class HumanPlayer : public Player {
protected:
public:
  HumanPlayer(const std::string &nickname, int init_lp, PlayerId duel_player,
              bool verbose = false)
      : Player(nickname, init_lp, duel_player, verbose) {}

  int think(const std::vector<std::string> &options) override {
    while (true) {
      std::string input = getline();
      if (input == "quit") {
        exit(0);
      }
      auto it = std::find(options.begin(), options.end(), input);
      if (it != options.end()) {
        return std::distance(options.begin(), it);
      } else {
        fmt::println("{} Choose from {}", duel_player_, options);
      }
    }
  }
};

class YGOProEnvFns {
public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("deck1"_.Bind(std::string("OldSchool")),
                    "deck2"_.Bind(std::string("OldSchool")), "player"_.Bind(-1),
                    "play_mode"_.Bind(std::string("bot")),
                    "verbose"_.Bind(false), "max_options"_.Bind(16),
                    "max_cards"_.Bind(80), "n_history_actions"_.Bind(16),
                    "record"_.Bind(false), "async_reset"_.Bind(true), "greedy_reward"_.Bind(true));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config &conf) {
    int n_action_feats = 13;
    return MakeDict(
        "obs:cards_"_.Bind(Spec<uint8_t>({conf["max_cards"_] * 2, 41})),
        "obs:global_"_.Bind(Spec<uint8_t>({23})),
        "obs:actions_"_.Bind(
            Spec<uint8_t>({conf["max_options"_], n_action_feats})),
        "obs:h_actions_"_.Bind(
            Spec<uint8_t>({conf["n_history_actions"_], n_action_feats + 2})),
        "info:num_options"_.Bind(Spec<int>({}, {0, conf["max_options"_] - 1})),
        "info:to_play"_.Bind(Spec<int>({}, {0, 1})),
        "info:is_selfplay"_.Bind(Spec<int>({}, {0, 1})),
        "info:win_reason"_.Bind(Spec<int>({}, {-1, 1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config &conf) {
    return MakeDict(
        "action"_.Bind(Spec<int>({}, {0, conf["max_options"_] - 1})));
  }
};

using YGOProEnvSpec = EnvSpec<YGOProEnvFns>;

enum PlayMode { kHuman, kSelfPlay, kRandomBot, kGreedyBot, kCount };

// parse play modes seperated by '+'
inline std::vector<PlayMode> parse_play_modes(const std::string &play_mode) {
  std::vector<PlayMode> modes;
  std::istringstream ss(play_mode);
  std::string token;
  while (std::getline(ss, token, '+')) {
    if (token == "human") {
      modes.push_back(kHuman);
    } else if (token == "self") {
      modes.push_back(kSelfPlay);
    } else if (token == "bot") {
      modes.push_back(kGreedyBot);
    } else if (token == "random") {
      modes.push_back(kRandomBot);
    } else {
      throw std::runtime_error("Unknown play mode: " + token);
    }
  }
  // human mode can't be combined with other modes
  if (std::find(modes.begin(), modes.end(), kHuman) != modes.end() &&
      modes.size() > 1) {
    throw std::runtime_error("Human mode can't be combined with other modes");
  }
  return modes;
}

// rules = 1, Traditional
// rules = 0, Default
// rules = 4, Link
// rules = 5, MR5
constexpr int32_t rules_ = 5;
constexpr int32_t duel_options_ = ((rules_ & 0xFF) << 16) + (0 & 0xFFFF);


class YGOProEnv : public Env<YGOProEnvSpec> {
protected:
  constexpr static int init_lp_ = 8000;
  constexpr static int startcount_ = 5;
  constexpr static int drawcount_ = 1;

  std::string deck1_;
  std::string deck2_;
  std::vector<uint32> main_deck0_;
  std::vector<uint32> main_deck1_;
  std::vector<uint32> extra_deck0_;
  std::vector<uint32> extra_deck1_;

  std::string deck_name_[2] = {"", ""};
  std::string nickname_[2] = {"Alice", "Bob"};

  const std::vector<PlayMode> play_modes_;

  // if play_mode_ == 'bot' or 'human', player_ is the order of the ai player
  // -1 means random, 0 and 1 means the first and second player respectively
  const int player_;

  PlayMode play_mode_;
  bool verbose_ = false;

  int max_episode_steps_, elapsed_step_;

  PlayerId ai_player_;

  intptr_t pduel_ = 0;
  Player *players_[2]; //  abstract class must be pointer

  std::uniform_int_distribution<uint64_t> dist_int_;
  bool done_{true};
  bool duel_started_{false};
  uint32_t eng_flag_{0};

  PlayerId winner_;
  uint8_t win_reason_;
  bool greedy_reward_;

  int lp_[2];

  // turn player
  PlayerId tp_;
  int current_phase_;
  int turn_count_;

  int msg_;
  std::vector<std::string> options_;
  PlayerId to_play_;
  std::function<void(int)> callback_;

  byte data_[4096];
  int dp_ = 0;
  int dl_ = 0;

  byte query_buf_[4096];
  int qdp_ = 0;

  byte resp_buf_[128];

  using IdleCardSpec = std::tuple<CardCode, std::string, uint32_t>;

  // chain
  PlayerId chaining_player_;

  double step_time_ = 0;
  uint64_t step_time_count_ = 0;

  double reset_time_ = 0;
  double reset_time_1_ = 0;
  double reset_time_2_ = 0;
  double reset_time_3_ = 0;
  uint64_t reset_time_count_ = 0;

  const int n_history_actions_;

  // circular buffer for history actions
  TArray<uint8_t> history_actions_;
  int ha_p_ = 0;
  std::vector<CardId> h_card_ids_;

  std::unordered_set<std::string> revealed_;

  // multi select
  int ms_idx_ = -1;
  int ms_mode_ = 0;
  int ms_min_ = 0;
  int ms_max_ = 0;
  int ms_must_ = 0;
  std::vector<std::string> ms_specs_;
  std::vector<std::vector<int>> ms_combs_;
  ankerl::unordered_dense::map<std::string, int> ms_spec2idx_;
  std::vector<int> ms_r_idxs_;

  // discard hand cards
  bool discard_hand_ = false;

  // replay
  bool record_ = false;
  FILE* fp_ = nullptr;
  bool is_recording = false;

  // MSG_SELECT_COUNTER
  int n_counters_ = 0;

  // async reset
  const bool async_reset_;
  int n_lives_ = 0;
  std::future<MDuel> duel_fut_;
  BS::thread_pool pool_;
  std::mt19937 duel_gen_;


public:
  YGOProEnv(const Spec &spec, int env_id)
      : Env<YGOProEnvSpec>(spec, env_id),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1), dist_int_(0, 0xffffffff),
        deck1_(spec.config["deck1"_]), deck2_(spec.config["deck2"_]),
        player_(spec.config["player"_]), players_{nullptr, nullptr},
        play_modes_(parse_play_modes(spec.config["play_mode"_])),
        verbose_(spec.config["verbose"_]), record_(spec.config["record"_]),
        n_history_actions_(spec.config["n_history_actions"_]), pool_(BS::thread_pool(1)),
        async_reset_(spec.config["async_reset"_]), greedy_reward_(spec.config["greedy_reward"_]) {
    if (record_) {
      if (!verbose_) {
        throw std::runtime_error("record mode must be used with verbose mode and num_envs=1");
      }
    }

    duel_gen_ = std::mt19937(dist_int_(gen_));

    if (async_reset_) {
      duel_fut_ = pool_.submit_task([
        this, duel_seed=dist_int_(gen_)] {
        return new_duel(duel_seed);
      });
    }

    int max_options = spec.config["max_options"_];
    int n_action_feats = spec.state_spec["obs:actions_"_].shape[1];
    h_card_ids_.resize(max_options);
    history_actions_ = TArray<uint8_t>(Array(
        ShapeSpec(sizeof(uint8_t), {n_history_actions_, n_action_feats + 2})));
  }

  ~YGOProEnv() {
    for (int i = 0; i < 2; i++) {
      if (players_[i] != nullptr) {
        delete players_[i];
      }
    }
  }

  int max_options() const { return spec_.config["max_options"_]; }

  int max_cards() const { return spec_.config["max_cards"_]; }

  bool IsDone() override { return done_; }

  bool random_mode() const { return play_modes_.size() > 1; }

  bool self_play() const {
    return std::find(play_modes_.begin(), play_modes_.end(), kSelfPlay) !=
           play_modes_.end();
  }

  void update_time_stat(const clock_t& start, uint64_t time_count, double& time_stat) {
    double seconds = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    time_stat = time_stat * (static_cast<double>(time_count) /
      (time_count + 1)) + seconds / (time_count + 1);
  }

  MDuel new_duel(uint32_t seed) {
    auto pduel = YGO_CreateDuel(seed);
    MDuel mduel{pduel, seed};

    for (PlayerId i = 0; i < 2; i++) {
      YGO_SetPlayerInfo(pduel, i, init_lp_, startcount_, drawcount_);
      auto [main_deck, extra_deck, deck_name] = load_deck(pduel, i, duel_gen_);
      if (i == 0) {
        mduel.main_deck0 = main_deck;
        mduel.extra_deck0 = extra_deck;
        mduel.deck_name0 = deck_name;
      } else {
        mduel.main_deck1 = main_deck;
        mduel.extra_deck1 = extra_deck;
        mduel.deck_name1 = deck_name;
      }
    }
    YGO_StartDuel(pduel, duel_options_);
    return mduel;
  }

  void Reset() override {
    clock_t start = clock();

    if (random_mode()) {
      play_mode_ = play_modes_[dist_int_(gen_) % play_modes_.size()];
    } else {
      play_mode_ = play_modes_[0];
    }

    if (play_mode_ != kSelfPlay) {
      if (player_ == -1) {
        ai_player_ = dist_int_(gen_) % 2;
      } else {
        ai_player_ = player_;
      }
    }

    turn_count_ = 0;
    ms_idx_ = -1;

    history_actions_.Zero();
    ha_p_ = 0;

    clock_t _start = clock();

    intptr_t old_duel = pduel_;
    MDuel mduel;
    if (async_reset_) {
      mduel = duel_fut_.get();
      n_lives_ = 1;
    } else {
      mduel = new_duel(dist_int_(gen_));
    }

    auto duel_seed = mduel.seed;
    pduel_ = mduel.pduel;

    deck_name_[0] = mduel.deck_name0;
    deck_name_[1] = mduel.deck_name1;
    main_deck0_ = mduel.main_deck0;
    extra_deck0_ = mduel.extra_deck0;
    main_deck1_ = mduel.main_deck1;
    extra_deck1_ = mduel.extra_deck1;

    for (PlayerId i = 0; i < 2; i++) {
      if (players_[i] != nullptr) {
        delete players_[i];
      }
      std::string nickname = i == 0 ? "Alice" : "Bob";
      if (i == ai_player_) {
        nickname = "Agent";
      }
      nickname_[i] = nickname;
      if ((play_mode_ == kHuman) && (i != ai_player_)) {
        players_[i] = new HumanPlayer(nickname_[i], init_lp_, i, verbose_);
      } else if (play_mode_ == kRandomBot) {
        players_[i] = new RandomAI(max_options(), dist_int_(gen_), nickname_[i],
                                   init_lp_, i, verbose_);
      } else {
        players_[i] = new GreedyAI(nickname_[i], init_lp_, i, verbose_);
      }
      lp_[i] = players_[i]->init_lp_;
    }

    if (record_) {
      if (is_recording && fp_ != nullptr) {
        fclose(fp_);
      }
      auto time_str = time_now();
      // Use last 4 digits of seed as unique id
      auto seed_ = duel_seed % 10000;
      std::string fname;
      while (true) {
        fname = fmt::format("./replay/a{} {:04d}.yrp", time_str, seed_);
        // check existence
        if (std::filesystem::exists(fname)) {
          seed_ = (seed_ + 1) % 10000;
        } else {
          break;
        } 
      }
      fp_ = fopen(fname.c_str(), "wb");
      if (!fp_) {
        throw std::runtime_error("Failed to open file for replay: " + fname);
      }

      is_recording = true;

      ReplayHeader rh;
      rh.id = 0x31707279;
      rh.version = 0x00001360;
      rh.flag = REPLAY_UNIFORM;
      rh.seed = duel_seed;
      rh.start_time = (unsigned int)time(nullptr);
      fwrite(&rh, sizeof(rh), 1, fp_);

      for (PlayerId i = 0; i < 2; i++) {
        uint16_t name[20];
        memset(name, 0, 40);
        std::string name_str = fmt::format("{} {}", nickname_[i], deck_name_[i]);
        if (name_str.size() > 20) {
          // truncate
          name_str = name_str.substr(0, 20);
        }
        fmt::println("name: {}", name_str);
        str_to_uint16(name_str.c_str(), name);
        fwrite(name, 40, 1, fp_);
      }

      ReplayWriteInt32(init_lp_);
      ReplayWriteInt32(startcount_);
      ReplayWriteInt32(drawcount_);
      ReplayWriteInt32(duel_options_);

      for (PlayerId i = 0; i < 2; i++) {
        auto &main_deck = i == 0 ? main_deck0_ : main_deck1_;
        auto &extra_deck = i == 0 ? extra_deck0_ : extra_deck1_;
        ReplayWriteInt32(main_deck.size());
        for (auto code : main_deck) {
          ReplayWriteInt32(code);
        }
        ReplayWriteInt32(extra_deck.size());
        for (int j = int(extra_deck.size()) - 1; j >= 0; --j) {
          ReplayWriteInt32(extra_deck[j]);
        }
      }

    }

    duel_started_ = true;
    winner_ = 255;
    win_reason_ = 255;

    // update_time_stat(_start, reset_time_count_, reset_time_2_);
    // _start = clock();

    next();

    done_ = false;
    elapsed_step_ = 0;
    WriteState(0.0);

    if (async_reset_) {
      duel_fut_ = pool_.submit_task([
        this, old_duel, duel_seed=dist_int_(gen_)] {
        if (old_duel != 0) {
          YGO_EndDuel(old_duel);
        }
        return new_duel(duel_seed);
      });
    }
    // update_time_stat(_start, reset_time_count_, reset_time_3_);

    // update_time_stat(start, reset_time_count_, reset_time_);
    // reset_time_count_++;
    // if (reset_time_count_ % 20 == 0) {
    //   fmt::println("Reset time: {:.3f}, {:.3f}, {:.3f}", reset_time_ * 1000, reset_time_2_ * 1000, reset_time_3_ * 1000);
    // }
  }

  void init_multi_select(
    int min, int max, int must, const std::vector<std::string> &specs,
    int mode = 0, const std::vector<std::vector<int>> &combs = {}) {
    ms_idx_ = 0;
    ms_mode_ = mode;
    ms_min_ = min;
    ms_max_ = max;
    ms_must_ = must;
    ms_specs_ = specs;
    ms_r_idxs_.clear();
    ms_spec2idx_.clear();

    for (int j = 0; j < ms_specs_.size(); ++j) {
      const auto &spec = ms_specs_[j];
      ms_spec2idx_[spec] = j;
    }

    if (ms_mode_ == 0) {
      for (int j = 0; j < ms_specs_.size(); ++j) {
        const auto &spec = ms_specs_[j];
        options_.push_back(spec);
      }
    } else {
      ms_combs_ = combs;
      _callback_multi_select_2_prepare();
    }
  }

  void handle_multi_select() {
    options_ = {};
    if (ms_mode_ == 0) {
      for (int j = 0; j < ms_specs_.size(); ++j) {
        if (ms_spec2idx_.find(ms_specs_[j]) != ms_spec2idx_.end()) {
          options_.push_back(ms_specs_[j]);
        }
      }
      if (ms_idx_ == ms_max_ - 1) {
        if (ms_idx_ >= ms_min_) {
          options_.push_back("f");
        }
        callback_ = [this](int idx) {
          _callback_multi_select(idx, true);
        };
      } else if (ms_idx_ >= ms_min_) {
        options_.push_back("f");
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };
      } else {
        callback_ = [this](int idx) {
          _callback_multi_select(idx, false);
        };    
      }
    } else {
      _callback_multi_select_2_prepare();
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };
    }
  }

  int get_ms_spec_idx(const std::string &spec) const {
    auto it = ms_spec2idx_.find(spec);
    if (it != ms_spec2idx_.end()) {
      return it->second;
    }
    // TODO: find the root cause
    // print ms_spec2idx
    show_deck(0);
    show_deck(1);
    show_buffer();
    show_turn();
    fmt::println("MS: idx: {}, mode: {}, min: {}, max: {}, must: {}, specs: {}, combs: {}, r_idx: {}", ms_idx_, ms_mode_, ms_min_, ms_max_, ms_must_, ms_specs_, ms_combs_, ms_r_idxs_);
    fmt::print("ms_spec2idx: ");
    for (const auto &[k, v] : ms_spec2idx_) {
      fmt::print("({}, {}), ", k, v);
    }
    fmt::print("\n");
    return -1;
    // throw std::runtime_error("Spec not found: " + spec);
  }

  void _callback_multi_select_2(int idx) {
    const auto &option = options_[idx];
    idx = get_ms_spec_idx(option);
    if (idx == -1) {
      // TODO: find the root cause
      fmt::println("options: {}, idx: {}, option: {}", options_, idx, option);
      throw std::runtime_error("Spec not found");
    }
    ms_r_idxs_.push_back(idx);
    std::vector<std::vector<int>> combs;
    for (auto &c : ms_combs_) {
      if (c[0] == idx) {
        c.erase(c.begin());
        if (c.empty()) {
          _callback_multi_select_2_finish();
          return;
        } else {
          combs.push_back(c);
        }
      }
    }
    ms_idx_++;
    ms_combs_ = combs;
  }

  void _callback_multi_select_2_prepare() {
    std::set<int> comb;
    for (const auto &c : ms_combs_) {
      comb.insert(c[0]);
    }
    for (auto &i : comb) {
      const auto &spec = ms_specs_[i];
      options_.push_back(spec);
    }
  }

  void _callback_multi_select_2_finish() {
    ms_idx_ = -1;
    resp_buf_[0] = ms_r_idxs_.size() + ms_must_;
    for (int i = 0; i < ms_must_; ++i) {
      resp_buf_[i + 1] = 0;
    }
    for (int i = 0; i < ms_r_idxs_.size(); ++i) {
      resp_buf_[i + ms_must_ + 1] = ms_r_idxs_[i];
    }
    YGO_SetResponseb(pduel_, resp_buf_);
  }

  void _callback_multi_select(int idx, bool finish) {
    const auto &option = options_[idx];
    // fmt::println("Select card: {}, finish: {}", option, finish);
    if (option == "f") {
      finish = true;
    } else {
      idx = get_ms_spec_idx(option);
      if (idx != -1) {
        ms_r_idxs_.push_back(idx);
      } else {
        // TODO: find the root cause
        fmt::println("options: {}, idx: {}, option: {}", options_, idx, option);
        ms_idx_ = -1;
        resp_buf_[0] = ms_min_;
        for (int i = 0; i < ms_min_; ++i) {
          resp_buf_[i + 1] = i;
        }
        YGO_SetResponseb(pduel_, resp_buf_);
        return;
      }
    }
    if (finish) {
      ms_idx_ = -1;
      resp_buf_[0] = ms_r_idxs_.size();
      for (int i = 0; i < ms_r_idxs_.size(); ++i) {
        resp_buf_[i + 1] = ms_r_idxs_[i];
      }
      YGO_SetResponseb(pduel_, resp_buf_);
    } else {
      ms_idx_++;
      ms_spec2idx_.erase(option);
    }
  }

  void update_h_card_ids(PlayerId player, int idx) {
    h_card_ids_[idx] = parse_card_id(options_[idx], player);
  }

  void update_history_actions(PlayerId player, int idx) {
    if ((msg_ == MSG_SELECT_CHAIN) & (options_[idx][0] == 'c')) {
      return;
    }
    ha_p_--;
    if (ha_p_ < 0) {
      ha_p_ = n_history_actions_ - 1;
    }
    history_actions_[ha_p_].Zero();
    _set_obs_action(history_actions_, ha_p_, msg_, options_[idx], {},
                    h_card_ids_[idx]);
    history_actions_[ha_p_](13) = static_cast<uint8_t>(player);
    history_actions_[ha_p_](14) = static_cast<uint8_t>(turn_count_);
  }

  void show_deck(const std::vector<CardCode> &deck, const std::string &prefix) const {
    fmt::print("{} deck: [", prefix);
    for (int i = 0; i < deck.size(); i++) {
      fmt::print(" '{}'", c_get_card(deck[i]).name());
    }
    fmt::print(" ]\n");
  }

  void show_turn() const {
    fmt::println("turn: {}, phase: {}, tplayer: {}", turn_count_, phase_to_string(current_phase_), tp_);
  }

  void show_buffer() const {
    fmt::println("msg: {}, dp: {}, dl: {}", msg_to_string(msg_), dp_, dl_);
    for (int i = 0; i < dl_; ++i) {
      fmt::print("{:02x} ", data_[i]);
    }
    fmt::print("\n");
  }

  void show_deck(PlayerId player) const {
    fmt::print("Player {}'s deck: {}\n", player, deck_name_[player]);
    // show_deck(player == 0 ? main_deck0_ : main_deck1_, "Main");
    // show_deck(player == 0 ? extra_deck0_ : extra_deck1_, "Extra");
  }

  void show_history_actions(PlayerId player) const {
    const auto &ha = history_actions_;
    // print card ids of history actions
    for (int i = 0; i < n_history_actions_; ++i) {
      fmt::print("history {}\n", i);
      uint8_t msg_id = uint8_t(ha(i, 2));
      int msg = _msgs[msg_id - 1];
      fmt::print("msg: {},", msg_to_string(msg));
      uint8_t v1 = ha(i, 0);
      uint8_t v2 = ha(i, 1);
      CardId card_id = (static_cast<CardId>(v1) << 8) + static_cast<CardId>(v2);
      fmt::print(" {};", card_id);
      for (int j = 3; j < ha.Shape()[1]; j++) {
        fmt::print(" {}", uint8_t(ha(i, j)));
      }
      fmt::print("\n");
    }
  }

  void Step(const Action &action) override {
    // clock_t start = clock();

    int idx = action["action"_];
    callback_(idx);
    update_history_actions(to_play_, idx);

    PlayerId player = to_play_;

    if (verbose_) {
      show_decision(idx);
    }

    if (ms_idx_ != -1) {
      handle_multi_select();
    } else {
      next();
    }

    float reward = 0;
    int reason = 0;
    if (done_) {
      float base_reward;
      if (greedy_reward_) {
        if (winner_ == 0) {
          if (turn_count_ <= 1) {
            // FTK
            base_reward = 16.0;
          } else if (turn_count_ <= 3) {
            base_reward = 8.0;
          } else if (turn_count_ <= 5) {
            base_reward = 4.0;
          } else if (turn_count_ <= 7) {
            base_reward = 2.0;
          } else {
            base_reward = 0.5 + 1.0 / (turn_count_ - 7);
          }
        } else {
          if (turn_count_ <= 1) {
            base_reward = 8.0;
          } else if (turn_count_ <= 3) {
            base_reward = 4.0;
          } else if (turn_count_ <= 5) {
            base_reward = 2.0;
          } else {
            base_reward = 0.5 + 1.0 / (turn_count_ - 5);
          }
        }
      } else {
        base_reward = 1.0;
      }

      if (play_mode_ == kSelfPlay) {
        // to_play_ is the previous player
        reward = winner_ == to_play_ ? base_reward : -base_reward;
      } else {
        reward = winner_ == ai_player_ ? base_reward : -base_reward;
      }

      if (win_reason_ == 0x01) {
        reason = 1;
      } else if (win_reason_ == 0x02) {
        reason = -1;
      }

      if (record_) {
        if (!is_recording || fp_ == nullptr) {
          throw std::runtime_error("Recording is not started");
        }
        fclose(fp_);
        is_recording = false;
      }
    }

    WriteState(reward, win_reason_);

    // update_time_stat(start, step_time_count_, step_time_);
    // step_time_count_++;
    // if (step_time_count_ % 3000 == 0) {
    //   fmt::println("Step time: {:.3f}", step_time_ * 1000);
    // }
  }

private:
  using SpecIndex = ankerl::unordered_dense::map<std::string, uint16_t>;

  std::tuple<SpecIndex, std::vector<int>> _set_obs_cards(TArray<uint8_t> &f_cards, PlayerId to_play) {
    SpecIndex spec2index;
    std::vector<int> loc_n_cards;
    int offset = 0;
    for (auto pi = 0; pi < 2; pi++) {
      const PlayerId player = (to_play + pi) % 2;
      const bool opponent = pi == 1;
      std::vector<std::pair<uint8_t, bool>> configs = {
          {LOCATION_DECK, true},   {LOCATION_HAND, true},
          {LOCATION_MZONE, false}, {LOCATION_SZONE, false},
          {LOCATION_GRAVE, false}, {LOCATION_REMOVED, false},
          {LOCATION_EXTRA, true},
      };
      for (auto &[location, hidden_for_opponent] : configs) {
        // check this
        if (opponent && (revealed_.size() != 0)) {
          hidden_for_opponent = false;
        }
        if (opponent && hidden_for_opponent) {
          auto n_cards = YGO_QueryFieldCount(pduel_, player, location);
          loc_n_cards.push_back(n_cards);
          for (auto i = 0; i < n_cards; i++) {
            f_cards(offset, 2) = location_to_id(location);
            f_cards(offset, 4) = 1;
            offset++;
          }
        } else {
          std::vector<Card> cards = get_cards_in_location(player, location);
          int n_cards = cards.size();
          loc_n_cards.push_back(n_cards);
          for (int i = 0; i < n_cards; ++i) {
            const auto &c = cards[i];
            auto spec = c.get_spec(opponent);
            bool hide = false;
            if (opponent) {
              hide = c.position_ & POS_FACEDOWN;
              if (revealed_.find(spec) != revealed_.end()) {
                hide = false;
              }
            }
            _set_obs_card_(f_cards, offset, c, hide);
            offset++;
            spec2index[spec] = static_cast<uint16_t>(offset);
          }
        }
      }
    }
    return {spec2index, loc_n_cards};
  }

  void _set_obs_card_(TArray<uint8_t> &f_cards, int offset, const Card &c,
                      bool hide) {
    // check offset exceeds max_cards
    uint8_t location = c.location_;
    bool overlay = location & LOCATION_OVERLAY;
    if (overlay) {
      location = location & 0x7f;
    }
    if (overlay) {
      hide = false;
    }

    if (!hide) {
      auto card_id = c_get_card_id(c.code_);
      f_cards(offset, 0) = static_cast<uint8_t>(card_id >> 8);
      f_cards(offset, 1) = static_cast<uint8_t>(card_id & 0xff);
    }
    f_cards(offset, 2) = location_to_id(location);

    uint8_t seq = 0;
    if (location == LOCATION_MZONE || location == LOCATION_SZONE ||
        location == LOCATION_GRAVE) {
      seq = c.sequence_ + 1;
    }
    f_cards(offset, 3) = seq;
    f_cards(offset, 4) = (c.controler_ != to_play_) ? 1 : 0;
    if (overlay) {
      f_cards(offset, 5) = position_to_id(POS_FACEUP);
      f_cards(offset, 6) = 1;
    } else {
      if (location == LOCATION_DECK || location == LOCATION_HAND || location == LOCATION_EXTRA) {
        if (hide || (c.position_ & POS_FACEDOWN)) {
          f_cards(offset, 5) = position_to_id(POS_FACEDOWN);
        }
        // else {
        //   fmt::println("location: {}, position: {}", location2str.at(location), position_to_string(c.position_));
        // }
      } else {
        f_cards(offset, 5) = position_to_id(c.position_);
      }
    }
    if (!hide) {
      f_cards(offset, 7) = attribute_to_id(c.attribute_);
      f_cards(offset, 8) = race_to_id(c.race_);
      f_cards(offset, 9) = c.level_;
      f_cards(offset, 10) = std::min(c.counter_, static_cast<uint32_t>(15));
      f_cards(offset, 11) = static_cast<uint8_t>((c.status_ & (STATUS_DISABLED | STATUS_FORBIDDEN)) != 0);
      auto [atk1, atk2] = float_transform(c.attack_);
      f_cards(offset, 12) = atk1;
      f_cards(offset, 13) = atk2;

      auto [def1, def2] = float_transform(c.defense_);
      f_cards(offset, 14) = def1;
      f_cards(offset, 15) = def2;

      auto type_ids = type_to_ids(c.type_);
      for (int j = 0; j < type_ids.size(); ++j) {
        f_cards(offset, 16 + j) = type_ids[j];
      }
    }
  }

  void _set_obs_global(TArray<uint8_t> &feat, PlayerId player, const std::vector<int> &loc_n_cards) {
    uint8_t me = player;
    uint8_t op = 1 - player;

    auto [me_lp_1, me_lp_2] = float_transform(lp_[me]);
    feat(0) = me_lp_1;
    feat(1) = me_lp_2;

    auto [op_lp_1, op_lp_2] = float_transform(lp_[op]);
    feat(2) = op_lp_1;
    feat(3) = op_lp_2;

    feat(4) = std::min(turn_count_, 16);
    feat(5) = phase_to_id(current_phase_);
    feat(6) = (me == 0) ? 1 : 0;
    feat(7) = (me == tp_) ? 1 : 0;

    for (int i = 0; i < loc_n_cards.size(); i++) {
      feat(8 + i) = static_cast<uint8_t>(loc_n_cards[i]);
    }
  }

  void _set_obs_action_spec(TArray<uint8_t> &feat, int i,
                            const std::string &spec,
                            const SpecIndex &spec2index,
                            CardId card_id = 0) {
    uint16_t idx;
    if (spec2index.empty()) {
      idx = card_id;
    } else {
      auto it = spec2index.find(spec);
      if (it == spec2index.end()) {
        // TODO: find the root cause
        // print spec2index
        show_deck(0);
        show_deck(1);
        show_buffer();
        show_turn();
        fmt::println("MS: idx: {}, mode: {}, min: {}, max: {}, must: {}, specs: {}, combs: {}", ms_idx_, ms_mode_, ms_min_, ms_max_, ms_must_, ms_specs_, ms_combs_);
        fmt::println("Spec: {}, Spec2index:", spec);
        for (auto &[k, v] : spec2index) {
          fmt::print("{}: {}, ", k, v);
        }
        fmt::print("\n");
        // throw std::runtime_error("Spec not found: " + spec);
        idx = 1;
      } else {
        idx = it->second;
      }
    }
    feat(i, 0) = static_cast<uint8_t>(idx >> 8);
    feat(i, 1) = static_cast<uint8_t>(idx & 0xff);
  }

  void _set_obs_action_msg(TArray<uint8_t> &feat, int i, int msg) {
    feat(i, 2) = msg_to_id(msg);
  }

  void _set_obs_action_act(TArray<uint8_t> &feat, int i, char act,
                           uint8_t act_offset = 0) {
    feat(i, 3) = cmd_act_to_id(act) + act_offset;
  }

  void _set_obs_action_yesno(TArray<uint8_t> &feat, int i, char yesno) {
    feat(i, 4) = cmd_yesno_to_id(yesno);
  }

  void _set_obs_action_phase(TArray<uint8_t> &feat, int i, char phase) {
    feat(i, 5) = cmd_phase_to_id(phase);
  }

  void _set_obs_action_cancel(TArray<uint8_t> &feat, int i) {
    feat(i, 6) = 1;
  }

  void _set_obs_action_finish(TArray<uint8_t> &feat, int i) {
    feat(i, 7) = 1;
  }

  void _set_obs_action_position(TArray<uint8_t> &feat, int i, char position) {
    position = 1 << (position - '1');
    feat(i, 8) = position_to_id(position);
  }

  void _set_obs_action_option(TArray<uint8_t> &feat, int i, char option) {
    feat(i, 9) = option - '0';
  }

  void _set_obs_action_number(TArray<uint8_t> &feat, int i, char number) {
    feat(i, 10) = number - '0';
  }

  void _set_obs_action_place(TArray<uint8_t> &feat, int i, const std::string &spec) {
    feat(i, 11) = cmd_place_to_id(spec);
  }

  void _set_obs_action_attrib(TArray<uint8_t> &feat, int i, uint8_t attrib) {
    feat(i, 12) = attribute_to_id(attrib);
  }

  void _set_obs_action(TArray<uint8_t> &feat, int i, int msg,
                       const std::string &option, const SpecIndex &spec2index,
                       CardId card_id) {
    _set_obs_action_msg(feat, i, msg);
    if (msg == MSG_SELECT_IDLECMD) {
      if (option == "b" || option == "e") {
        _set_obs_action_phase(feat, i, option[0]);
      } else {
        auto act = option[0];
        auto spec = option.substr(2);
        uint8_t offset = 0;
        int n = spec.size();
        if (act == 'v' && std::isalpha(spec[n - 1])) {
          offset = spec[n - 1] - 'a';
          spec = spec.substr(0, n - 1);
        }
        _set_obs_action_act(feat, i, act, offset);

        _set_obs_action_spec(feat, i, spec, spec2index, card_id);
      }
    } else if (msg == MSG_SELECT_CHAIN) {
      if (option[0] == 'c') {
        _set_obs_action_cancel(feat, i);
      } else {
        char act = 'v';
        auto spec = option;
        uint8_t offset = 0;
        auto n = spec.size();
        if (std::isalpha(spec[n - 1])) {
          offset = spec[n - 1] - 'a';
          spec = spec.substr(0, n - 1);
        }
        _set_obs_action_act(feat, i, act, offset);

        _set_obs_action_spec(feat, i, spec, spec2index, card_id);
      }
    } else if (msg == MSG_SELECT_CARD || msg == MSG_SELECT_TRIBUTE ||
               msg == MSG_SELECT_SUM || msg == MSG_SELECT_UNSELECT_CARD) {
      if (option[0] == 'f') {
        _set_obs_action_finish(feat, i);
      } else {
        _set_obs_action_spec(feat, i, option, spec2index, card_id);
      }
    } else if (msg == MSG_SELECT_POSITION) {
      _set_obs_action_position(feat, i, option[0]);
    } else if (msg == MSG_SELECT_EFFECTYN) {
      auto spec = option.substr(2);
      _set_obs_action_spec(feat, i, spec, spec2index, card_id);

      _set_obs_action_yesno(feat, i, option[0]);
    } else if (msg == MSG_SELECT_YESNO) {
      _set_obs_action_yesno(feat, i, option[0]);
    } else if (msg == MSG_SELECT_BATTLECMD) {
      if (option == "m" || option == "e") {
        _set_obs_action_phase(feat, i, option[0]);
      } else {
        auto act = option[0];
        auto spec = option.substr(2);
        _set_obs_action_act(feat, i, act);
        _set_obs_action_spec(feat, i, spec, spec2index, card_id);
      }
    } else if (msg == MSG_SELECT_OPTION) {
      _set_obs_action_option(feat, i, option[0]);
    } else if (msg == MSG_SELECT_PLACE || msg_ == MSG_SELECT_DISFIELD) {
      _set_obs_action_place(feat, i, option);
    } else if (msg == MSG_ANNOUNCE_ATTRIB) {
      _set_obs_action_attrib(feat, i, 1 << (option[0] - '1'));
    } else if (msg == MSG_ANNOUNCE_NUMBER) {
      _set_obs_action_number(feat, i, option[0]);
    } else {
      throw std::runtime_error("Unsupported message " + std::to_string(msg));
    }
  }

  CardId spec_to_card_id(const std::string &spec, PlayerId player) {
    int offset = 0;
    // TODO: possible info leak
    if (spec[0] == 'o') {
      player = 1 - player;
      offset++;
    }
    auto [loc, seq, pos] = spec_to_ls(spec.substr(offset));
    return c_get_card_id(get_card_code(player, loc, seq));
  }

  CardId parse_card_id(const std::string &option, PlayerId player) {
    CardId card_id = 0;
    if (msg_ == MSG_SELECT_IDLECMD) {
      if (!(option == "b" || option == "e")) {
        auto n = option.size();
        if (std::isalpha(option[n - 1])) {
          card_id = spec_to_card_id(option.substr(2, n - 3), player);
        } else {
          card_id = spec_to_card_id(option.substr(2), player);
        }
      }
    } else if (msg_ == MSG_SELECT_CHAIN) {
      if (option != "c") {
        card_id = spec_to_card_id(option, player);
      }
    } else if (msg_ == MSG_SELECT_CARD || msg_ == MSG_SELECT_TRIBUTE ||
               msg_ == MSG_SELECT_SUM || msg_ == MSG_SELECT_UNSELECT_CARD) {
      if (option[0] != 'f') {
        card_id = spec_to_card_id(option, player);
      }
    } else if (msg_ == MSG_SELECT_EFFECTYN) {
      card_id = spec_to_card_id(option.substr(2), player);
    } else if (msg_ == MSG_SELECT_BATTLECMD) {
      if (!(option == "m" || option == "e")) {
        card_id = spec_to_card_id(option.substr(2), player);
      }
    }
    return card_id;
  }

  void _set_obs_actions(TArray<uint8_t> &feat, const SpecIndex &spec2index,
                        int msg, const std::vector<std::string> &options) {
    for (int i = 0; i < options.size(); ++i) {
      _set_obs_action(feat, i, msg, options[i], spec2index, 0);
    }
  }


  void str_to_uint16(const char* src, uint16_t* dest) {
      for (int i = 0; i < strlen(src); i += 1) {
        dest[i] = src[i];
      }

      // Add null terminator
      dest[strlen(src) + 1] = '\0';
  }

  void ReplayWriteInt8(int8_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  void ReplayWriteInt32(int32_t value) {
    fwrite(&value, sizeof(value), 1, fp_);
  }

  // ygopro-core API
  intptr_t YGO_CreateDuel(uint32_t seed) {
    std::mt19937 rnd(seed);
    // return create_duel(rnd());
    duel* pduel = new duel();
    pduel->random.reset(rnd());
    return (intptr_t)pduel;
  }

  void YGO_SetPlayerInfo(intptr_t pduel, int32 playerid, int32 lp, int32 startcount, int32 drawcount) const {
    set_player_info(pduel, playerid, lp, startcount, drawcount);
  }

  void YGO_NewCard(intptr_t pduel, uint32 code, uint8 owner, uint8 playerid, uint8 location, uint8 sequence, uint8 position) const {
    new_card(pduel, code, owner, playerid, location, sequence, position);
  }

  void YGO_StartDuel(intptr_t pduel, int32 options) const {
    start_duel(pduel, options);
  }

  void YGO_EndDuel(intptr_t pduel) const {
    // end_duel(pduel);
    duel* pd = (duel*)pduel;
    delete pd;
  }

  int32 YGO_GetMessage(intptr_t pduel, byte* buf) {
    return get_message(pduel, buf);
  }

  uint32 YGO_Process(intptr_t pduel) {
    return process(pduel);
  }

  int32 YGO_QueryCard(intptr_t pduel, uint8 playerid, uint8 location, uint8 sequence, int32 query_flag, byte* buf) {
    return query_card(pduel, playerid, location, sequence, query_flag, buf, 0);
  }

  int32 YGO_QueryFieldCount(intptr_t pduel, uint8 playerid, uint8 location) {
    return query_field_count(pduel, playerid, location);
  }

  int32 YGO_QueryFieldCard(intptr_t pduel, uint8 playerid, uint8 location, uint32 query_flag, byte* buf) {
    return query_field_card(pduel, playerid, location, query_flag, buf, 0);
  }

  void YGO_SetResponsei(intptr_t pduel, int32 value) {
    if (record_) {
      ReplayWriteInt8(4);
      ReplayWriteInt32(value);
    }
    set_responsei(pduel, value);
  }

  void YGO_SetResponseb(intptr_t pduel, byte* buf) {
    if (record_) {
      switch (msg_) {
        case MSG_SORT_CARD:
          ReplayWriteInt8(1);
          fwrite(buf, 1, 1, fp_);
          break;
        case MSG_SELECT_COUNTER:
          ReplayWriteInt8(2 * n_counters_);
          fwrite(buf, 2 * n_counters_, 1, fp_);
          break;
        case MSG_SELECT_PLACE:
        case MSG_SELECT_DISFIELD:
          ReplayWriteInt8(3);
          fwrite(buf, 3, 1, fp_);
          break;
        default:
          ReplayWriteInt8(buf[0] + 1);
          fwrite(buf, buf[0] + 1, 1, fp_);
          break;
      }
    }
    set_responseb(pduel, buf);
  }

  // ygopro-core API

  void WriteState(float reward, int win_reason = 0) {
    State state = Allocate();

    int n_options = options_.size();
    state["reward"_] = reward;
    state["info:to_play"_] = int(to_play_);
    state["info:is_selfplay"_] = int(play_mode_ == kSelfPlay);
    state["info:win_reason"_] = win_reason;

    if (n_options == 0) {
      state["info:num_options"_] = 1;
      state["obs:global_"_][22] = uint8_t(1);
      return;
    }

    auto [spec2index, loc_n_cards] = _set_obs_cards(state["obs:cards_"_], to_play_);

    _set_obs_global(state["obs:global_"_], to_play_, loc_n_cards);

    // we can't shuffle because idx must be stable in callback
    if (n_options > max_options()) {
      options_.resize(max_options());
    }

    // print spec2index
    // for (auto const& [key, val] : spec2index) {
    //   fmt::println("{} {}", key, val);
    // }

    _set_obs_actions(state["obs:actions_"_], spec2index, msg_, options_);

    n_options = options_.size();
    state["info:num_options"_] = n_options;

    // update_h_card_ids from state
    for (int i = 0; i < n_options; ++i) {
      uint8_t spec_index1 = state["obs:actions_"_](i, 0);
      uint8_t spec_index2 = state["obs:actions_"_](i, 1);
      uint16_t spec_index = (static_cast<uint16_t>(spec_index1) << 8) + static_cast<uint16_t>(spec_index2);
      if (spec_index == 0) {
        h_card_ids_[i] = 0;
      } else {
        uint8_t card_id1 = state["obs:cards_"_](spec_index - 1, 0);
        uint8_t card_id2 = state["obs:cards_"_](spec_index - 1, 1);
        h_card_ids_[i] = (static_cast<uint16_t>(card_id1) << 8) + static_cast<uint16_t>(card_id2);
      }
    }

    // write history actions

    int offset = n_history_actions_ - ha_p_;
    int n_h_action_feats = history_actions_.Shape()[1];

    state["obs:h_actions_"_].Assign(
      (uint8_t *)history_actions_[ha_p_].Data(), n_h_action_feats * offset);
    state["obs:h_actions_"_][offset].Assign(
      (uint8_t *)history_actions_.Data(), n_h_action_feats * ha_p_);
    
    for (int i = 0; i < n_history_actions_; ++i) {
      if (uint8_t(state["obs:h_actions_"_](i, 2)) == 0) {
        break;
      }
      state["obs:h_actions_"_](i, 13) = static_cast<uint8_t>(uint8_t(state["obs:h_actions_"_](i, 13)) == to_play_);
      int turn_diff = std::min(16, turn_count_ - uint8_t(state["obs:h_actions_"_](i, 14)));
      state["obs:h_actions_"_](i, 14) = static_cast<uint8_t>(turn_diff);
    }
  }

  void show_decision(int idx) {
    fmt::println("Player {} chose \"{}\" in {}", to_play_, options_[idx],
                 options_);
  }

  std::tuple<std::vector<CardCode>, std::vector<CardCode>, std::string>
  load_deck(
    intptr_t pduel, PlayerId player, std::mt19937& gen, bool shuffle = true) const {
    std::string deck_name = player == 0 ? deck1_ : deck2_;

    if (deck_name == "random") {
      // generate random deck name
      std::uniform_int_distribution<uint64_t> dist_int(0,
                                                       deck_names_.size() - 1);
      deck_name = deck_names_[dist_int(gen)];
    }

    std::vector<CardCode> main_deck = main_decks_.at(deck_name);
    std::vector<CardCode> extra_deck = extra_decks_.at(deck_name);

    if (verbose_) {
      fmt::println("{} {}: {}, main({}), extra({})", player, nickname_[player],
        deck_name, main_deck.size(), extra_deck.size());
    }

    if (shuffle) {
      std::shuffle(main_deck.begin(), main_deck.end(), gen);
    }

    // add main deck in reverse order following ygopro
    // but since we have shuffled deck, so just add in order

    for (int i = 0; i < main_deck.size(); i++) {
      YGO_NewCard(pduel, main_deck[i], player, player, LOCATION_DECK, 0,
               POS_FACEDOWN_DEFENSE);
    }

    // add extra deck in reverse order following ygopro
    for (int i = int(extra_deck.size()) - 1; i >= 0; --i) {
      YGO_NewCard(pduel, extra_deck[i], player, player, LOCATION_EXTRA, 0,
               POS_FACEDOWN_DEFENSE);
    }

    return {main_deck, extra_deck, deck_name};
  }

  void next() {
    while (duel_started_) {
      if (eng_flag_ == PROCESSOR_END) {
        break;
      }
      uint32_t res = YGO_Process(pduel_);
      dl_ = res & PROCESSOR_BUFFER_LEN;
      eng_flag_ = res & PROCESSOR_FLAG;

      if (dl_ == 0) {
        continue;
      }
      YGO_GetMessage(pduel_, data_);
      dp_ = 0;
      while ((dp_ != dl_) || (ms_idx_ != -1)) {
        if (ms_idx_ != -1) {
          handle_multi_select();
        } else {
          handle_message();
          if (options_.empty()) {
            continue;
          }
        }
        if ((play_mode_ == kSelfPlay) || (to_play_ == ai_player_)) {
          if (options_.size() == 1) {
            callback_(0);
            update_h_card_ids(to_play_, 0);
            update_history_actions(to_play_, 0);
            if (verbose_) {
              show_decision(0);
            }
          } else {
            return;
          }
        } else {
          auto idx = players_[to_play_]->think(options_);
          callback_(idx);
          if (verbose_) {
            show_decision(idx);
          }
        }
      }
    }
    done_ = true;
    options_.clear();
  }

  uint8_t read_u8() { return data_[dp_++]; }

  uint16_t read_u16() {
    uint16_t v = *reinterpret_cast<uint16_t *>(data_ + dp_);
    dp_ += 2;
    return v;
  }

  uint32 read_u32() {
    uint32 v = *reinterpret_cast<uint32_t *>(data_ + dp_);
    dp_ += 4;
    return v;
  }

  uint32 q_read_u8() {
    uint8_t v = *reinterpret_cast<uint8_t *>(query_buf_ + qdp_);
    qdp_ += 1;
    return v;
  }

  uint32 q_read_u32() {
    uint32_t v = *reinterpret_cast<uint32_t *>(query_buf_ + qdp_);
    qdp_ += 4;
    return v;
  }

  CardCode get_card_code(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE;
    int32_t bl = YGO_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      throw std::runtime_error("[get_card_code] Invalid card");
    }
    qdp_ += 8;
    return q_read_u32();
  }

  Card get_card(PlayerId player, uint8_t loc, uint8_t seq) {
    int32_t flags = QUERY_CODE | QUERY_ATTACK | QUERY_DEFENSE | QUERY_POSITION |
                    QUERY_LEVEL | QUERY_RANK | QUERY_LSCALE | QUERY_RSCALE |
                    QUERY_LINK;
    int32_t bl = YGO_QueryCard(pduel_, player, loc, seq, flags, query_buf_);
    qdp_ = 0;
    if (bl <= 0) {
      throw std::runtime_error("[get_card] Invalid card (bl <= 0)");
    }
    uint32_t f = q_read_u32();
    if (f == LEN_EMPTY) {
      return Card();
    }
    f = q_read_u32();
    CardCode code = q_read_u32();
    Card c = c_get_card(code);
    uint32_t position = q_read_u32();
    c.set_location(position);
    uint32_t level = q_read_u32();
    if ((level & 0xff) > 0) {
      c.level_ = level & 0xff;
    }
    uint32_t rank = q_read_u32();
    if ((rank & 0xff) > 0) {
      c.level_ = rank & 0xff;
    }
    c.attack_ = q_read_u32();
    c.defense_ = q_read_u32();
    c.lscale_ = q_read_u32();
    c.rscale_ = q_read_u32();
    uint32_t link = q_read_u32();
    uint32_t link_marker = q_read_u32();
    if ((link & 0xff) > 0) {
      c.level_ = link & 0xff;
    }
    if (link_marker > 0) {
      c.defense_ = link_marker;
    }
    return c;
  }

  std::vector<Card> get_cards_in_location(PlayerId player, uint8_t loc) {
    int32_t flags = QUERY_CODE | QUERY_POSITION | QUERY_LEVEL | QUERY_RANK |
                    QUERY_ATTACK | QUERY_DEFENSE | QUERY_EQUIP_CARD |
                    QUERY_OVERLAY_CARD | QUERY_COUNTERS | QUERY_STATUS |
                    QUERY_LSCALE | QUERY_RSCALE | QUERY_LINK;
    int32_t bl = YGO_QueryFieldCard(pduel_, player, loc, flags, query_buf_);
    qdp_ = 0;
    std::vector<Card> cards;
    while (true) {
      if (qdp_ >= bl) {
        break;
      }
      uint32_t f = q_read_u32();
      if (f == LEN_EMPTY) {
        continue;
        ;
      }
      f = q_read_u32();
      CardCode code = q_read_u32();
      Card c = c_get_card(code);

      uint8_t controller = q_read_u8();
      uint8_t location = q_read_u8();
      uint8_t sequence = q_read_u8();
      uint8_t position = q_read_u8();
      c.controler_ = controller;
      c.location_ = location;
      c.sequence_ = sequence;
      c.position_ = position;

      uint32_t level = q_read_u32();
      if ((level & 0xff) > 0) {
        c.level_ = level & 0xff;
      }
      uint32_t rank = q_read_u32();
      if ((rank & 0xff) > 0) {
        c.level_ = rank & 0xff;
      }
      c.attack_ = q_read_u32();
      c.defense_ = q_read_u32();

      // TODO: equip_target
      if (f & QUERY_EQUIP_CARD) {
        q_read_u32();
      }

      uint32_t n_xyz = q_read_u32();
      for (int i = 0; i < n_xyz; ++i) {
        auto code = q_read_u32();
        Card c_ = c_get_card(code);
        c_.controler_ = controller;
        c_.location_ = location | LOCATION_OVERLAY;
        c_.sequence_ = sequence;
        c_.position_ = i;
        cards.push_back(c_);
      }

      // TODO: counters
      uint32_t n_counters = q_read_u32();
      for (int i = 0; i < n_counters; ++i) {
        if (i == 0) {
          c.counter_ = q_read_u32();
        }
        else {
          q_read_u32();
        }
      }

      c.status_ = q_read_u32();
      c.lscale_ = q_read_u32();
      c.rscale_ = q_read_u32();

      uint32_t link = q_read_u32();
      uint32_t link_marker = q_read_u32();
      if ((link & 0xff) > 0) {
        c.level_ = link & 0xff;
      }
      if (link_marker > 0) {
        c.defense_ = link_marker;
      }
      cards.push_back(c);
    }
    return cards;
  }

  std::vector<Card> read_cardlist(bool extra = false, bool extra8 = false) {
    std::vector<Card> cards;
    auto count = read_u8();
    cards.reserve(count);
    for (int i = 0; i < count; ++i) {
      auto code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto card = get_card(controller, loc, seq);
      if (extra) {
        if (extra8) {
          card.data_ = read_u8();
        } else {
          card.data_ = read_u32();
        }
      }
      cards.push_back(card);
    }
    return cards;
  }

  std::vector<IdleCardSpec> read_cardlist_spec(PlayerId player, bool extra = false, bool extra8 = false) {
    std::vector<IdleCardSpec> card_specs;
    auto count = read_u8();
    card_specs.reserve(count);
    for (int i = 0; i < count; ++i) {
      CardCode code = read_u32();
      auto controller = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      uint32_t data = -1;
      if (extra) {
        if (extra8) {
          data = read_u8();
        } else {
          data = read_u32();
        }
      }
      card_specs.push_back({code, ls_to_spec(loc, seq, 0, player != controller), data});
    }
    return card_specs;
  }

  std::string cardlist_info_for_player(const Card &card, PlayerId pl) {
    std::string spec = card.get_spec(pl);
    if (card.location_ == LOCATION_DECK) {
      spec = "deck";
    }
    if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
      return position_to_string(card.position_) + "card (" + spec + ")";
    }
    return card.name_ + " (" + spec + ")";
  }

  // This function does the following:
  // 1. read msg_ from data_ and update dp_
  // 2. (optional) print information if verbose_ is true
  // 3. update to_play_ and options_ if need action
  void handle_message() {
    msg_ = int(data_[dp_++]);
    options_ = {};

    if (verbose_) {
      fmt::println("Message {}, length {}, dp {}", msg_to_string(msg_), dl_, dp_);
    }

    if (msg_ == MSG_DRAW) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto drawed = read_u8();
      std::vector<uint32> codes;
      for (int i = 0; i < drawed; ++i) {
        uint32 code = read_u32();
        codes.push_back(code & 0x7fffffff);
      }
      const auto &pl = players_[player];
      pl->notify(fmt::format("Drew {} cards:", drawed));
      for (int i = 0; i < drawed; ++i) {
        const auto &c = c_get_card(codes[i]);
        pl->notify(fmt::format("{}: {}", i + 1, c.name_));
      }
      const auto &op = players_[1 - player];
      op->notify(fmt::format("Opponent drew {} cards.", drawed));
    } else if (msg_ == MSG_NEW_TURN) {
      tp_ = int(read_u8());
      turn_count_++;
      if (!verbose_) {
        return;
      }
      auto player = players_[tp_];
      player->notify("Your turn.");
      players_[1 - tp_]->notify(fmt::format("{}'s turn.", player->nickname_));
    } else if (msg_ == MSG_NEW_PHASE) {
      current_phase_ = int(read_u16());
      if (!verbose_) {
        return;
      }
      auto phase_str = phase_to_string(current_phase_);
      for (int i = 0; i < 2; ++i) {
        players_[i]->notify(fmt::format("Entering {} phase.", phase_str));
      }
    } else if (msg_ == MSG_MOVE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      uint32_t location = read_u32();
      uint32_t newloc = read_u32();
      uint32_t reason = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);
      Card cnew = c_get_card(code);
      cnew.set_location(newloc);
      auto pl = players_[card.controler_];
      auto op = players_[1 - card.controler_];

      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto plnewspec = cnew.get_spec(false);
      auto opnewspec = cnew.get_spec(true);

      auto getspec = [&](Player *p) { return p == pl ? plspec : opspec; };
      auto getnewspec = [&](Player *p) {
        return p == pl ? plnewspec : opnewspec;
      };
      bool card_visible = true;
      if ((card.position_ & POS_FACEDOWN) && (cnew.position_ & POS_FACEDOWN)) {
        card_visible = false;
      }
      auto getvisiblename = [&](Player *p) {
        return card_visible ? card.name_ : "Face-down card";
      };

      if ((reason & REASON_DESTROY) && (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("Card {} ({}) destroyed.", plspec, card.name_));
        op->notify(fmt::format("Card {} ({}) destroyed.", opspec, card.name_));
      } else if ((card.location_ == cnew.location_) &&
                 (card.location_ & LOCATION_ONFIELD)) {
        if (card.controler_ != cnew.controler_) {
          pl->notify(
              fmt::format("Your card {} ({}) changed controller to {} and is "
                          "now located at {}.",
                          plspec, card.name_, op->nickname_, plnewspec));
          op->notify(
              fmt::format("You now control {}'s card {} ({}) and it's located "
                          "at {}.",
                          pl->nickname_, opspec, card.name_, opnewspec));
        } else {
          pl->notify(fmt::format("Your card {} ({}) switched its zone to {}.",
                                 plspec, card.name_, plnewspec));
          op->notify(fmt::format("{}'s card {} ({}) switched its zone to {}.",
                                 pl->nickname_, opspec, card.name_, opnewspec));
        }
      } else if ((reason & REASON_DISCARD) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You discarded {} ({})", plspec, card.name_));
        op->notify(fmt::format("{} discarded {} ({})", pl->nickname_, opspec,
                               card.name_));
      } else if ((card.location_ == LOCATION_REMOVED) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your banished card {} ({}) returns to the field at "
                        "{}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s banished card {} ({}) returns to the field at "
                        "{}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((card.location_ == LOCATION_GRAVE) &&
                 (cnew.location_ & LOCATION_ONFIELD)) {
        pl->notify(
            fmt::format("Your card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        plspec, card.name_, plnewspec));
        op->notify(
            fmt::format("{}'s card {} ({}) returns from the graveyard to the "
                        "field at {}.",
                        pl->nickname_, opspec, card.name_, opnewspec));
      } else if ((cnew.location_ == LOCATION_HAND) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(
            fmt::format("Card {} ({}) returned to hand.", plspec, card.name_));
      } else if ((reason & (REASON_RELEASE | REASON_SUMMON)) &&
                 (card.location_ != cnew.location_)) {
        pl->notify(fmt::format("You tribute {} ({}).", plspec, card.name_));
        op->notify(fmt::format("{} tributes {} ({}).", pl->nickname_, opspec,
                               getvisiblename(op)));
      } else if ((card.location_ == (LOCATION_OVERLAY | LOCATION_MZONE)) &&
                 (cnew.location_ & LOCATION_GRAVE)) {
        pl->notify(fmt::format("You detached {}.", card.name_));
        op->notify(fmt::format("{} detached {}.", pl->nickname_, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_GRAVE)) {
        pl->notify(fmt::format("Your card {} ({}) was sent to the graveyard.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) was sent to the graveyard.",
                               pl->nickname_, opspec, card.name_));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_REMOVED)) {
        pl->notify(
            fmt::format("Your card {} ({}) was banished.", plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) was banished.", pl->nickname_,
                               opspec, getvisiblename(op)));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_DECK)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your deck.",
                               plspec, card.name_));
        op->notify(fmt::format("{}'s card {} ({}) returned to their deck.",
                               pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ != cnew.location_) &&
                 (cnew.location_ == LOCATION_EXTRA)) {
        pl->notify(fmt::format("Your card {} ({}) returned to your extra deck.",
                               plspec, card.name_));
        op->notify(
            fmt::format("{}'s card {} ({}) returned to their extra deck.",
                        pl->nickname_, opspec, getvisiblename(op)));
      } else if ((card.location_ == LOCATION_DECK) &&
                 (cnew.location_ == LOCATION_SZONE) &&
                 (cnew.position_ != POS_FACEDOWN)) {
        pl->notify(fmt::format("Activating {} ({})", plnewspec, card.name_));
        op->notify(fmt::format("{} activating {} ({})", pl->nickname_, opspec,
                               cnew.name_));
      }
    } else if (msg_ == MSG_SWAP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code1 = read_u32();
      uint32_t loc1 = read_u32();
      CardCode code2 = read_u32();
      uint32_t loc2 = read_u32();
      Card cards[2];
      cards[0] = c_get_card(code1);
      cards[1] = c_get_card(code2);
      cards[0].set_location(loc1);
      cards[1].set_location(loc2);

      for (PlayerId pl = 0; pl < 2; pl++) {
        for (int i = 0; i < 2; i++) {
          auto c = cards[i];
          auto spec = c.get_spec(pl);
          auto plname = players_[1 - c.controler_]->nickname_;
          players_[pl]->notify("Card " + c.name_ + " swapped control towards " +
                               plname + " and is now located at " + spec + ".");
        }
      }
    } else if (msg_ == MSG_SET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      uint32_t location = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);
      auto c = card.controler_;
      auto cpl = players_[c];
      auto opl = players_[1 - c];
      cpl->notify(fmt::format("You set {} ({}) in {} position.", card.name_,
                              card.get_spec(c), card.get_position()));
      opl->notify(fmt::format("{} sets {} in {} position.", cpl->nickname_,
                              card.get_spec(PlayerId(1 - c)),
                              card.get_position()));
    } else if (msg_ == MSG_EQUIP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto c = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto pos = read_u8();
      Card card = get_card(c, loc, seq);
      c = read_u8();
      loc = read_u8();
      seq = read_u8();
      pos = read_u8();
      Card target = get_card(c, loc, seq);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto c = cardlist_info_for_player(card, pl);
        auto t = cardlist_info_for_player(target, pl);
        players_[pl]->notify(fmt::format("{} equipped to {}.", c, t));
      }
    } else if (msg_ == MSG_HINT) {
      auto hint_type = read_u8();
      auto player = read_u8();
      auto value = read_u32();

      if (hint_type == HINT_SELECTMSG && value == 501) {
        discard_hand_ = true;
      }
      // non-GUI don't need hint
      return;
      if (hint_type == HINT_SELECTMSG) {
        if (value > 2000) {
          CardCode code = value;
          players_[player]->notify(fmt::format("{} select {}",
                                               players_[player]->nickname_,
                                               c_get_card(code).name_));
        } else {
          players_[player]->notify(get_system_string(value));
        }
      } else if (hint_type == HINT_NUMBER) {
        players_[1 - player]->notify(
            fmt::format("Choice of player: {}", value));
      } else {
        fmt::println("Unknown hint type {} with value {}", hint_type, value);
      }
    } else if (msg_ == MSG_CARD_HINT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      uint8_t player = read_u8();
      uint8_t loc = read_u8();
      uint8_t seq = read_u8();
      uint8_t pos = read_u8();
      uint8_t type = read_u8();
      uint32_t value = read_u32();
      Card card = get_card(player, loc, seq);
      if (card.code_ == 0) {
        return;
      }
      if (type == CHINT_RACE) {
        std::string races_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           races_str));
        }
      } else if (type == CHINT_ATTRIBUTE) {
        std::string attributes_str = "TODO";
        for (PlayerId pl = 0; pl < 2; pl++) {
          players_[pl]->notify(fmt::format("{} ({}) selected {}.",
                                           card.get_spec(pl), card.name_,
                                           attributes_str));
        }
      } else {
        fmt::println("Unknown card hint type {} with value {}", type, value);
      }
    } else if (msg_ == MSG_POS_CHANGE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      uint8_t prevpos = card.position_;
      card.position_ = read_u8();

      auto pl = players_[card.controler_];
      auto op = players_[1 - card.controler_];
      auto plspec = card.get_spec(false);
      auto opspec = card.get_spec(true);
      auto prevpos_str = position_to_string(prevpos);
      auto pos_str = position_to_string(card.position_);
      pl->notify("The position of card " + plspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
      op->notify("The position of card " + opspec + " (" + card.name_ +
                 ") changed from " + prevpos_str + " to " + pos_str + ".");
    } else if (msg_ == MSG_BECOME_TARGET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto u = read_u8();
      uint32_t target = read_u32();
      uint8_t tc = target & 0xff;
      uint8_t tl = (target >> 8) & 0xff;
      uint8_t tseq = (target >> 16) & 0xff;
      Card card = get_card(tc, tl, tseq);
      auto name = players_[chaining_player_]->nickname_;
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        auto tcname = card.name_;
        if ((card.controler_ != pl) && (card.position_ & POS_FACEDOWN)) {
          tcname = position_to_string(card.position_) + " card";
        }
        players_[pl]->notify(name + " targets " + spec + " (" + tcname + ")");
      }
    } else if (msg_ == MSG_CONFIRM_DECKTOP) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        cards.push_back(get_card(c, loc, seq));
      }

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto p = players_[pl];
        if (pl == player) {
          p->notify(fmt::format("You reveal {} cards from your deck:", size));
        } else {
          p->notify(fmt::format("{} reveals {} cards from their deck:",
                                players_[player]->nickname_, size));
        }
        for (int i = 0; i < size; ++i) {
          p->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
        }
      }
    } else if (msg_ == MSG_RANDOM_SELECTED) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto count = read_u8();
      std::vector<Card> cards;

      for (int i = 0; i < count; ++i) {
        auto c = read_u8();
        auto loc = read_u8();
        if (loc & LOCATION_OVERLAY) {
          throw std::runtime_error("Overlay not supported for random selected");
        }
        auto seq = read_u8();
        auto pos = read_u8();
        cards.push_back(get_card(c, loc, seq));
      }

      for (PlayerId pl = 0; pl < 2; pl++) {
        auto p = players_[pl];
        auto s = "card is";
        if (count > 1) {
          s = "cards are";
        }
        if (pl == player) {
          p->notify(fmt::format("Your {} {} randomly selected:", s, count));
        } else {
          p->notify(fmt::format("{}'s {} {} randomly selected:",
                                players_[player]->nickname_, s, count));
        }
        for (int i = 0; i < count; ++i) {
          p->notify(fmt::format("{}: {}", cards[i].get_spec(pl), cards[i].name_));
        }
      }
    } else if (msg_ == MSG_PLAYER_HINT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      dp_ += 6;
      // TODO: implement output
    } else if (msg_ == MSG_CARD_TARGET) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto c1 = read_u8();
      auto l1 = read_u8();
      auto s1 = read_u8();
      read_u8();
      auto c2 = read_u8();
      auto l2 = read_u8();
      auto s2 = read_u8();
      read_u8();

      Card card1 = get_card(c1, l1, s1);
      Card card2 = get_card(c2, l2, s2);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto p = players_[pl];
        auto spec1 = card1.get_spec(pl);
        auto spec2 = card2.get_spec(pl);
        auto c1name = card1.name_;
        auto c2name = card2.name_;
        if ((card1.controler_ != pl) && (card1.position_ & POS_FACEDOWN)) {
          c1name = position_to_string(card1.position_) + " card";
        }
        if ((card2.controler_ != pl) && (card2.position_ & POS_FACEDOWN)) {
          c2name = position_to_string(card2.position_) + " card";
        }
        p->notify(fmt::format(" {} ({}) targets {} ({})", spec1, c1name, spec2, c2name));
      }
    } else if (msg_ == MSG_CONFIRM_CARDS) {
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        if (verbose_) {
          cards.push_back(get_card(c, loc, seq));
        }
        revealed_.insert(ls_to_spec(loc, seq, 0, c == player));
      }
      if (!verbose_) {
        return;
      }

      auto pl = players_[player];
      auto op = players_[1 - player];

      op->notify(fmt::format("{} shows you {} cards.", pl->nickname_, size));
      for (int i = 0; i < size; ++i) {
        pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
      }
    } else if (msg_ == MSG_MISSED_EFFECT) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      dp_ += 4;
      CardCode code = read_u32();
      Card card = c_get_card(code);
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        auto str = get_system_string(1622);
        std::string fmt_str = "[%ls]";
        str = str.replace(str.find(fmt_str), fmt_str.length(), card.name_);
        players_[pl]->notify(str);
      }
    } else if (msg_ == MSG_SORT_CARD) {
      // TODO: implement action
      if (!verbose_) {
        dp_ = dl_;
        resp_buf_[0] = 255;
        YGO_SetResponseb(pduel_, resp_buf_);
        return;
      }
      auto player = read_u8();
      auto size = read_u8();
      std::vector<Card> cards;
      for (int i = 0; i < size; ++i) {
        read_u32();
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        cards.push_back(get_card(c, loc, seq));
      }
      auto pl = players_[player];
      pl->notify(
          "Sort " + std::to_string(size) +
          " cards by entering numbers separated by spaces (c = cancel):");
      for (int i = 0; i < size; ++i) {
        pl->notify(fmt::format("{}: {}", i + 1, cards[i].name_));
      }

      fmt::println("sort card action not implemented");
      resp_buf_[0] = 255;
      YGO_SetResponseb(pduel_, resp_buf_);

      // // generate all permutations
      // std::vector<int> perm(size);
      // std::iota(perm.begin(), perm.end(), 0);
      // std::vector<std::vector<int>> perms;
      // do {
      //   auto option = std::accumulate(perm.begin(), perm.end(),
      //   std::string(),
      //                                 [&](std::string &acc, int i) {
      //                                   return acc + std::to_string(i + 1) +
      //                                   " ";
      //                                 });
      //   options_.push_back(option);
      // } while (std::next_permutation(perm.begin(), perm.end()));
      // options_.push_back("c");
      // callback_ = [this](int idx) {
      //   const auto &option = options_[idx];
      //   if (option == "c") {
      //     resp_buf_[0] = 255;
      //     YGO_SetResponseb(pduel_, resp_buf_);
      //     return;
      //   }
      //   std::istringstream iss(option);
      //   int x;
      //   int i = 0;
      //   while (iss >> x) {
      //     resp_buf_[i] = uint8_t(x);
      //     i++;
      //   }
      //   YGO_SetResponseb(pduel_, resp_buf_);
      // };
    } else if (msg_ == MSG_ADD_COUNTER) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      auto c = get_card(player, loc, seq);
      auto pl = players_[player];
      PlayerId op_id = 1 - player;
      auto op = players_[op_id];
      // TODO: counter type to string
      pl->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} placed on {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_REMOVE_COUNTER) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto ctype = read_u16();
      auto player = read_u8();
      auto loc = read_u8();
      auto seq = read_u8();
      auto count = read_u16();
      auto c = get_card(player, loc, seq);
      auto pl = players_[player];
      PlayerId op_id = 1 - player;
      auto op = players_[op_id];
      pl->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(player)));
      op->notify(fmt::format("{} counter(s) of type {} removed from {} ().", count, "UNK", c.name_, c.get_spec(op_id)));
    } else if (msg_ == MSG_ATTACK_DISABLED) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      for (PlayerId pl = 0; pl < 2; pl++) {
        players_[pl]->notify(get_system_string(1621));
      }
    } else if (msg_ == MSG_SHUFFLE_SET_CARD) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      // TODO: implement output
      dp_ = dl_;
    } else if (msg_ == MSG_SHUFFLE_DECK) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto pl = players_[player];
      auto op = players_[1 - player];
      pl->notify("You shuffled your deck.");
      op->notify(pl->nickname_ + " shuffled their deck.");
    } else if (msg_ == MSG_SHUFFLE_EXTRA) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto player = read_u8();
      auto count = read_u8();
      for (int i = 0; i < count; ++i) {
        read_u32();
      }
      auto pl = players_[player];
      auto op = players_[1 - player];
      pl->notify(fmt::format("You shuffled your extra deck ({}).", count));
      op->notify(fmt::format("{} shuffled their extra deck ({}).", pl->nickname_, count));
    } else if (msg_ == MSG_SHUFFLE_HAND) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }

      auto player = read_u8();
      dp_ = dl_;

      auto pl = players_[player];
      auto op = players_[1 - player];
      pl->notify("You shuffled your hand.");
      op->notify(pl->nickname_ + " shuffled their hand.");
    } else if (msg_ == MSG_SUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_SUMMONING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      const auto &nickname = players_[card.controler_]->nickname_;
      for (auto pl : players_) {
        pl->notify(nickname + " summoning " + card.name_ + " (" +
                   std::to_string(card.attack_) + "/" +
                   std::to_string(card.defense_) + ") in " +
                   card.get_position() + " position.");
      }
    } else if (msg_ == MSG_SPSUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_FLIPSUMMONED) {
      dp_ = dl_;
    } else if (msg_ == MSG_FLIPSUMMONING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }

      auto code = read_u32();
      auto location = read_u32();
      Card card = c_get_card(code);
      card.set_location(location);

      auto cpl = players_[card.controler_];
      for (PlayerId pl = 0; pl < 2; pl++) {
        auto spec = card.get_spec(pl);
        players_[1 - pl]->notify(cpl->nickname_ + " flip summons " + spec +
                                 " (" + card.name_ + ")");
      }
    } else if (msg_ == MSG_SPSUMMONING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      const auto &nickname = players_[card.controler_]->nickname_;
      for (PlayerId p = 0; p < 2; p++) {
        auto pl = players_[p];
        auto pos = card.get_position();
        auto atk = std::to_string(card.attack_);
        auto def = std::to_string(card.defense_);
        std::string name = p == card.controler_ ? "You" : nickname;
        if (card.type_ & TYPE_LINK) {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + ") in " + pos + " position.");
        } else {
          pl->notify(name + " special summoning " + card.name_ + " (" +
                     atk + "/" + def + ") in " + pos + " position.");
        }
      }
    } else if (msg_ == MSG_CHAIN_NEGATED) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_DISABLED) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_SOLVED) {
      dp_ = dl_;
      revealed_.clear();
    } else if (msg_ == MSG_CHAIN_SOLVING) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINED) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAIN_END) {
      dp_ = dl_;
    } else if (msg_ == MSG_CHAINING) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      CardCode code = read_u32();
      Card card = c_get_card(code);
      card.set_location(read_u32());
      auto tc = read_u8();
      auto tl = read_u8();
      auto ts = read_u8();
      uint32_t desc = read_u32();
      auto cs = read_u8();
      auto c = card.controler_;
      PlayerId o = 1 - c;
      chaining_player_ = c;
      players_[c]->notify("Activating " + card.get_spec(c) + " (" + card.name_ +
                          ")");
      players_[o]->notify(players_[c]->nickname_ + " activating " +
                          card.get_spec(o) + " (" + card.name_ + ")");
    } else if (msg_ == MSG_DAMAGE) {
      auto player = read_u8();
      auto amount = read_u32();
      _damage(player, amount);
    } else if (msg_ == MSG_RECOVER) {
      auto player = read_u8();
      auto amount = read_u32();
      _recover(player, amount);
    } else if (msg_ == MSG_LPUPDATE) {
      auto player = read_u8();
      auto lp = read_u32();
      if (lp >= lp_[player]) {
        _recover(player, lp - lp_[player]);
      } else {
        _damage(player, lp_[player] - lp);
      }
    } else if (msg_ == MSG_PAY_LPCOST) {
      auto player = read_u8();
      auto cost = read_u32();
      lp_[player] -= cost;
      if (!verbose_) {
        return;
      }
      auto pl = players_[player];
      pl->notify("You pay " + std::to_string(cost) + " LP. Your LP is now " +
                 std::to_string(lp_[player]) + ".");
      players_[1 - player]->notify(
          pl->nickname_ + " pays " + std::to_string(cost) + " LP. " +
          pl->nickname_ + "'s LP is now " + std::to_string(lp_[player]) + ".");
    } else if (msg_ == MSG_ATTACK) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto attacker = read_u32();
      PlayerId ac = attacker & 0xff;
      auto aloc = (attacker >> 8) & 0xff;
      auto aseq = (attacker >> 16) & 0xff;
      auto apos = (attacker >> 24) & 0xff;
      auto target = read_u32();
      PlayerId tc = target & 0xff;
      auto tloc = (target >> 8) & 0xff;
      auto tseq = (target >> 16) & 0xff;
      auto tpos = (target >> 24) & 0xff;

      if ((ac == 0) && (aloc == 0) && (aseq == 0) && (apos == 0)) {
        return;
      }

      Card acard = get_card(ac, aloc, aseq);
      auto name = players_[ac]->nickname_;
      if ((tc == 0) && (tloc == 0) && (tseq == 0) && (tpos == 0)) {
        for (PlayerId i = 0; i < 2; i++) {
          players_[i]->notify(name + " prepares to attack with " +
                              acard.get_spec(i) + " (" + acard.name_ + ")");
        }
        return;
      }

      Card tcard = get_card(tc, tloc, tseq);
      for (PlayerId i = 0; i < 2; i++) {
        auto aspec = acard.get_spec(i);
        auto tspec = tcard.get_spec(i);
        auto tcname = tcard.name_;
        if ((tcard.controler_ != i) && (tcard.position_ & POS_FACEDOWN)) {
          tcname = tcard.get_position() + " card";
        }
        players_[i]->notify(name + " prepares to attack " + tspec + " (" +
                            tcname + ") with " + aspec + " (" + acard.name_ +
                            ")");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_START) {
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("begin damage");
      }
    } else if (msg_ == MSG_DAMAGE_STEP_END) {
      if (!verbose_) {
        return;
      }
      for (int i = 0; i < 2; i++) {
        players_[i]->notify("end damage");
      }
    } else if (msg_ == MSG_BATTLE) {
      if (!verbose_) {
        dp_ = dl_;
        return;
      }
      auto attacker = read_u32();
      auto aa = read_u32();
      auto ad = read_u32();
      auto bd0 = read_u8();
      auto target = read_u32();
      auto da = read_u32();
      auto dd = read_u32();
      auto bd1 = read_u8();

      auto ac = attacker & 0xff;
      auto aloc = (attacker >> 8) & 0xff;
      auto aseq = (attacker >> 16) & 0xff;

      auto tc = target & 0xff;
      auto tloc = (target >> 8) & 0xff;
      auto tseq = (target >> 16) & 0xff;
      auto tpos = (target >> 24) & 0xff;

      Card acard = get_card(ac, aloc, aseq);
      Card tcard;
      if (tloc != 0) {
        tcard = get_card(tc, tloc, tseq);
      }
      for (int i = 0; i < 2; i++) {
        auto pl = players_[i];
        std::string attacker_points;
        if (acard.type_ & TYPE_LINK) {
          attacker_points = std::to_string(aa);
        } else {
          attacker_points = std::to_string(aa) + "/" + std::to_string(ad);
        }
        if (tloc != 0) {
          std::string defender_points;
          if (tcard.type_ & TYPE_LINK) {
            defender_points = std::to_string(da);
          } else {
            defender_points = std::to_string(da) + "/" + std::to_string(dd);
          }
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks " +
                     tcard.name_ + " (" + defender_points + ")");
        } else {
          pl->notify(acard.name_ + "(" + attacker_points + ")" + " attacks");
        }
      }
    } else if (msg_ == MSG_WIN) {
      auto player = read_u8();
      auto reason = read_u8();
      auto winner = players_[player];
      auto loser = players_[1 - player];

      _duel_end(player, reason);

      auto l_reason = reason_to_string(reason);
      if (verbose_) {
        winner->notify("You won (" + l_reason + ").");
        loser->notify("You lost (" + l_reason + ").");
      }
    } else if (msg_ == MSG_RETRY) {
      throw std::runtime_error("Retry");
    } else if (msg_ == MSG_SELECT_BATTLECMD) {
      auto player = read_u8();
      auto activatable = read_cardlist_spec(player, true);
      auto attackable = read_cardlist_spec(player, true, true);
      bool to_m2 = read_u8();
      bool to_ep = read_u8();

      auto pl = players_[player];
      if (verbose_) {
        pl->notify("Battle menu:");
      }
      for (const auto [code, spec, data] : activatable) {
        // TODO: Add effect description to indicate which effect is being activated
        options_.push_back("v " + spec);
        if (verbose_) {
          auto [loc, seq, pos] = spec_to_ls(spec);
          auto c = get_card(player, loc, seq);
          pl->notify("v " + spec + ": activate " + c.name_ + " (" +
                     std::to_string(c.attack_) + "/" +
                     std::to_string(c.defense_) + ")");
        }
      }
      for (const auto [code, spec, data] : attackable) {
        // TODO: add this as feature
        bool direct_attackable = data & 0x1;
        options_.push_back("a " + spec);
        if (verbose_) {
          auto [loc, seq, pos] = spec_to_ls(spec);
          auto c = get_card(player, loc, seq);
          std::string s;
          if (c.type_ & TYPE_LINK) {
            s = "a " + spec + ": " + c.name_ + " (" +
                std::to_string(c.attack_) + ")";
          } else {
            s = "a " + spec + ": " + c.name_ + " (" +
                std::to_string(c.attack_) + "/" +
                std::to_string(c.defense_) + ")";
          }
          if (direct_attackable) {
            s += " direct attack";
          } else {
            s += " attack";
          }
          pl->notify(s);
        }
      }
      if (to_m2) {
        options_.push_back("m");
        if (verbose_) {
          pl->notify("m: Main phase 2.");
        }
      }
      if (to_ep) {
        if (!to_m2) {
          options_.push_back("e");
          if (verbose_) {
            pl->notify("e: End phase.");
          }
        }
      }
      int n_activatables = activatable.size();
      int n_attackables = attackable.size();
      to_play_ = player;
      callback_ = [this, n_activatables, n_attackables, to_ep, to_m2](int idx) {
        if (idx < n_activatables) {
          YGO_SetResponsei(pduel_, idx << 16);
        } else if (idx < (n_activatables + n_attackables)) {
          idx = idx - n_activatables;
          YGO_SetResponsei(pduel_, (idx << 16) + 1);
        } else if ((options_[idx] == "e") && to_ep) {
          YGO_SetResponsei(pduel_, 3);
        } else if ((options_[idx] == "m") && to_m2) {
          YGO_SetResponsei(pduel_, 2);
        } else {
          throw std::runtime_error("Invalid option");
        }
      };
    } else if (msg_ == MSG_SELECT_UNSELECT_CARD) {
      // TODO: add feature of selected cards (also for multi select)
      auto player = read_u8();
      bool finishable = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto select_size = read_u8();

      std::vector<std::string> select_specs;
      select_specs.reserve(select_size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < select_size; ++i) {
          auto code = read_u32();
          auto loc = read_u32();
          Card card = c_get_card(code);
          card.set_location(loc);
          cards.push_back(card);
        }
        auto pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          select_specs.push_back(spec);
          pl->notify(spec + ": " + card.name_);
        }
      } else {
        for (int i = 0; i < select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto pos = read_u8();
          auto spec = ls_to_spec(loc, seq, pos, controller != player);
          select_specs.push_back(spec);
        }
      }

      auto unselect_size = read_u8();

      // unselect not allowed (no regrets!)
      dp_ += 8 * unselect_size;

      for (int j = 0; j < select_specs.size(); ++j) {
        options_.push_back(select_specs[j]);
      }

      if (finishable) {
        options_.push_back("f");
      }

      // cancelable and finishable not needed

      to_play_ = player;
      callback_ = [this](int idx) {
        if (options_[idx] == "f") {
          YGO_SetResponsei(pduel_, -1);
        } else {
          resp_buf_[0] = 1;
          resp_buf_[1] = idx;
          YGO_SetResponseb(pduel_, resp_buf_);
        }
      };

    } else if (msg_ == MSG_SELECT_CARD) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto size = read_u8();

      if (min == 0) {
        throw std::runtime_error("Min == 0 not implemented for select card");
      }

      std::vector<std::string> specs;
      specs.reserve(size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < size; ++i) {
          auto code = read_u32();
          auto loc = read_u32();
          Card card = c_get_card(code);
          card.set_location(loc);
          cards.push_back(card);
        }
        auto pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) + " cards separated by spaces:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          specs.push_back(spec);
          if (card.controler_ != player && card.position_ & POS_FACEDOWN) {
            pl->notify(spec + ": " + card.get_position() + " card");
          } else {
            pl->notify(spec + ": " + card.name_);
          }
        }
      } else {
        for (int i = 0; i < size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto pos = read_u8();
          auto spec = ls_to_spec(loc, seq, pos, controller != player);
          specs.push_back(spec);
        }
      }

      if (discard_hand_) {
        discard_hand_ = false;
        if (current_phase_ == PHASE_END) {
          // random discard
          std::vector<int> comb(size);
          std::iota(comb.begin(), comb.end(), 0);
          std::shuffle(comb.begin(), comb.end(), gen_);
          resp_buf_[0] = min;
          for (int i = 0; i < min; ++i) {
            resp_buf_[i + 1] = comb[i];
          }
          YGO_SetResponseb(pduel_, resp_buf_);
          return;
        }
      }

      // TODO: use this when added to history actions
      // if ((min == max) && (max == specs.size())) {
      //   resp_buf_[0] = specs.size();
      //   for (int i = 0; i < specs.size(); ++i) {
      //     resp_buf_[i + 1] = i;
      //   }
      //   YGO_SetResponseb(pduel_, resp_buf_);
      //   return;
      // }

      init_multi_select(min, max, 0, specs);

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select(idx, ms_max_ == 1);
      };
    } else if (msg_ == MSG_SELECT_TRIBUTE) {
      auto player = read_u8();
      bool cancelable = read_u8();
      auto min = read_u8();
      auto max = read_u8();
      auto size = read_u8();

      if (min == 0) {
        throw std::runtime_error("Min == 0 not implemented for select tribute");
      }

      std::vector<int> release_params;
      release_params.reserve(size);
      std::vector<std::string> specs;
      specs.reserve(size);
      if (verbose_) {
        std::vector<Card> cards;
        for (int i = 0; i < size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto release_param = read_u8();
          Card card = get_card(controller, loc, seq);
          cards.push_back(card);
          release_params.push_back(release_param);
        }
        auto pl = players_[player];
        pl->notify("Select " + std::to_string(min) + " to " +
                   std::to_string(max) +
                   " cards to tribute separated by spaces:");
        for (const auto &card : cards) {
          auto spec = card.get_spec(player);
          specs.push_back(spec);
          pl->notify(spec + ": " + card.name_);
        }
      } else {
        for (int i = 0; i < size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto release_param = read_u8();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          specs.push_back(spec);

          release_params.push_back(release_param);
        }
      }

      bool has_weight =
          std::any_of(release_params.begin(), release_params.end(),
                      [](int i) { return i != 1; });

      if (min != max) {
        throw std::runtime_error(
          fmt::format("min({}) != max({}), not implemented for select tribute", min, max));
      }

      if (has_weight) {
        throw std::runtime_error("weight not implemented for select tribute");
        // combs = combinations_with_weight(release_params, min);
      }

      // TODO: use this when added to history actions
      // if (max == specs.size()) {
      //   // tribute all
      //   resp_buf_[0] = specs.size();
      //   for (int i = 0; i < specs.size(); ++i) {
      //     resp_buf_[i + 1] = i;
      //   }
      //   YGO_SetResponseb(pduel_, resp_buf_);
      //   return;
      // }

      init_multi_select(min, max, 0, specs);

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select(idx, ms_max_ == 1);
      };
    } else if (msg_ == MSG_SELECT_SUM) {
      // ritual summoning mode 1 (max)
      auto mode = read_u8();
      auto player = read_u8();
      auto val = read_u32();
      int _min = read_u8();
      int _max = read_u8();
      auto must_select_size = read_u8();

      if (mode == 0) {
        if (must_select_size > 2) {
          throw std::runtime_error(
              " must select size: " + std::to_string(must_select_size) +
              " not implemented for MSG_SELECT_SUM");
        }
      } else {
        throw std::runtime_error("mode: " + std::to_string(mode) +
                                 " not implemented for MSG_SELECT_SUM");
      }

      std::vector<int> select_params;
      std::vector<std::string> select_specs;

      int expected = val;
      if (verbose_) {
        std::vector<Card> must_select;
        must_select.reserve(must_select_size);
        for (int i = 0; i < must_select_size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();
          Card card = get_card(controller, loc, seq);
          must_select.push_back(card);
          expected -= (param & 0xff);
        }
        auto pl = players_[player];
        pl->notify("Select cards with a total value of " +
                   std::to_string(expected) + ", seperated by spaces.");
        for (const auto &card : must_select) {
          auto spec = card.get_spec(player);
          pl->notify(card.name_ + " (" + spec +
                     ") must be selected, automatically selected.");
        }
      } else {
        for (int i = 0; i < must_select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          expected -= (param & 0xff);
        }
      }

      uint8_t select_size = read_u8();
      select_params.reserve(select_size);
      select_specs.reserve(select_size);

      if (verbose_) {
        std::vector<Card> select;
        select.reserve(select_size);
        for (int i = 0; i < select_size; ++i) {
          auto code = read_u32();
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();
          Card card = get_card(controller, loc, seq);
          select.push_back(card);
          select_params.push_back(param);
        }
        auto pl = players_[player];
        for (const auto &card : select) {
          auto spec = card.get_spec(player);
          select_specs.push_back(spec);
          pl->notify(spec + ": " + card.name_);
        }
      } else {
        for (int i = 0; i < select_size; ++i) {
          dp_ += 4;
          auto controller = read_u8();
          auto loc = read_u8();
          auto seq = read_u8();
          auto param = read_u32();

          auto spec = ls_to_spec(loc, seq, 0, controller != player);
          select_specs.push_back(spec);
          select_params.push_back(param);
        }
      }

      std::vector<std::vector<int>> card_levels;
      for (int i = 0; i < select_size; ++i) {
        std::vector<int> levels;
        int level1 = select_params[i] & 0xff;
        int level2 = (select_params[i] >> 16);
        if (level1 > 0) {
          levels.push_back(level1);
        }
        if (level2 > 0) {
          levels.push_back(level2);
        }
        card_levels.push_back(levels);
      }

      // We assume any card_level can be the first

      std::vector<std::vector<int>> combs =
          combinations_with_weight2(card_levels, expected);
      
      for (auto &comb : combs) {
        std::sort(comb.begin(), comb.end());
      }

      init_multi_select(
        _min, _max, must_select_size, select_specs, 1, combs);

      to_play_ = player;
      callback_ = [this](int idx) {
        _callback_multi_select_2(idx);
      };

    } else if (msg_ == MSG_SELECT_CHAIN) {
      auto player = read_u8();
      auto size = read_u8();
      auto spe_count = read_u8();
      bool forced = read_u8();
      dp_ += 8;
      // auto hint_timing = read_u32();
      // auto other_timing = read_u32();

      std::vector<Card> cards;
      std::vector<uint32_t> descs;
      std::vector<uint32_t> spec_codes;
      for (int i = 0; i < size; ++i) {
        auto et = read_u8();
        CardCode code = read_u32();
        if (verbose_) {
          uint32_t loc = read_u32();
          Card card = c_get_card(code);
          card.set_location(loc);
          cards.push_back(card);
          spec_codes.push_back(card.get_spec_code(player));
        } else {
          PlayerId c = read_u8();
          uint8_t loc = read_u8();
          uint8_t seq = read_u8();
          uint8_t pos = read_u8();
          spec_codes.push_back(ls_to_spec_code(loc, seq, pos, c != player));
        }
        uint32_t desc = read_u32();
        descs.push_back(desc);
      }

      if ((size == 0) && (spe_count == 0)) {
        // non-GUI don't need this
        // if (verbose_) {
        //   fmt::println("keep processing");
        // }
        YGO_SetResponsei(pduel_, -1);
        return;
      }

      auto pl = players_[player];
      auto op = players_[1 - player];
      chaining_player_ = player;
      if (!op->seen_waiting_) {
        if (verbose_) {
          op->notify("Waiting for opponent.");
        }
        op->seen_waiting_ = true;
      }

      std::vector<int> chain_index;
      ankerl::unordered_dense::map<uint32_t, int> chain_counts;
      ankerl::unordered_dense::map<uint32_t, int> chain_orders;
      std::vector<std::string> chain_specs;
      std::vector<std::string> effect_descs;
      for (int i = 0; i < size; i++) {
        chain_index.push_back(i);
        chain_counts[spec_codes[i]] += 1;
      }
      for (int i = 0; i < size; i++) {
        auto spec_code = spec_codes[i];
        auto cs = code_to_spec(spec_code);
        auto chain_count = chain_counts[spec_code];
        if (chain_count > 1) {
          // TODO: should use desc to indicate activate which effect
          cs.push_back('a' + chain_orders[spec_code]);
        }
        chain_orders[spec_code]++;
        chain_specs.push_back(cs);
        if (verbose_) {
          const auto &card = cards[i];
          effect_descs.push_back(card.get_effect_description(descs[i], true));
        }
      }

      if (verbose_) {
        if (forced) {
          pl->notify("Select chain:");
        } else {
          pl->notify("Select chain (c to cancel):");
        }
        for (int i = 0; i < size; i++) {
          const auto &effect_desc = effect_descs[i];
          if (effect_desc.empty()) {
            pl->notify(chain_specs[i] + ": " + cards[i].name_);
          } else {
            pl->notify(chain_specs[i] + " (" + cards[i].name_ +
                       "): " + effect_desc);
          }
        }
      }

      for (const auto &spec : chain_specs) {
        options_.push_back(spec);
      }
      if (!forced) {
        options_.push_back("c");
      }
      to_play_ = player;
      callback_ = [this, forced](int idx) {
        const auto &option = options_[idx];
        if (option == "c") {
          if (forced) {
            fmt::print("cancel not allowed in forced chain\n");
            YGO_SetResponsei(pduel_, 0);
            return;
          }
          YGO_SetResponsei(pduel_, -1);
          return;
        }
        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_SELECT_YESNO) {
      auto player = read_u8();

      if (verbose_) {
        auto desc = read_u32();
        auto pl = players_[player];
        std::string opt;
        if (desc > 10000) {
          auto code = desc >> 4;
          auto card = c_get_card(code);
          auto opt_idx = desc & 0xf;
          if (opt_idx < card.strings_.size()) {
            opt = card.strings_[opt_idx];
          }
          if (opt.empty()) {
            opt = "Unknown question from " + card.name_ + ". Yes or no?";
          }
        } else {
          opt = get_system_string(desc);
        }
        pl->notify(opt);
        pl->notify("Please enter y or n.");
      } else {
        dp_ += 4;
      }
      options_ = {"y", "n"};
      to_play_ = player;
      callback_ = [this](int idx) {
        if (idx == 0) {
          YGO_SetResponsei(pduel_, 1);
        } else if (idx == 1) {
          YGO_SetResponsei(pduel_, 0);
        } else {
          throw std::runtime_error("Invalid option");
        }
      };
    } else if (msg_ == MSG_SELECT_EFFECTYN) {
      auto player = read_u8();

      std::string spec;
      if (verbose_) {
        CardCode code = read_u32();
        uint32_t loc = read_u32();
        Card card = c_get_card(code);
        card.set_location(loc);
        auto desc = read_u32();
        auto pl = players_[player];
        spec = card.get_spec(player);
        auto name = card.name_;
        std::string s;
        if (desc == 0) {
          // From [%ls], activate [%ls]?
          s = "From " + card.get_spec(player) + ", activate " + name + "?";
        } else if (desc < 2048) {
          s = get_system_string(desc);
          std::string fmt_str = "[%ls]";
          auto pos = find_substrs(s, fmt_str);
          if (pos.size() == 0) {
            // nothing to replace
          } else if (pos.size() == 1) {
            auto p = pos[0];
            s = s.substr(0, p) + name + s.substr(p + fmt_str.size());
          } else if (pos.size() == 2) {
            auto p1 = pos[0];
            auto p2 = pos[1];
            s = s.substr(0, p1) + card.get_spec(player) +
                s.substr(p1 + fmt_str.size(), p2 - p1 - fmt_str.size()) + name +
                s.substr(p2 + fmt_str.size());
          } else {
            throw std::runtime_error("Unknown effectyn desc " +
                                     std::to_string(desc) + " of " + name);
          }
        }  else if (desc < 10000u) {
          s = get_system_string(desc);
        } else {
          CardCode code = (desc >> 4) & 0x0fffffff;
          uint32_t offset = desc & 0xf;
          if (cards_.find(code) != cards_.end()) {
            auto &card_ = c_get_card(code);
            s = card_.strings_[offset];
            if (s.empty()) {
              s = "???";
            }
          } else {
            throw std::runtime_error("Unknown effectyn desc " +
                                     std::to_string(desc) + " of " + name);
          }
        }
        pl->notify(s);
        pl->notify("Please enter y or n.");
      } else {
        dp_ += 4;
        auto c = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        auto pos = read_u8();
        dp_ += 4;
        spec = ls_to_spec(loc, seq, pos, c != player);
      }
      options_ = {"y " + spec, "n " + spec};
      to_play_ = player;
      callback_ = [this](int idx) {
        if (idx == 0) {
          YGO_SetResponsei(pduel_, 1);
        } else if (idx == 1) {
          YGO_SetResponsei(pduel_, 0);
        } else {
          throw std::runtime_error("Invalid option");
        }
      };
    } else if (msg_ == MSG_SELECT_OPTION) {
      // TODO: add card information
      auto player = read_u8();
      auto size = read_u8();
      if (verbose_) {
        auto pl = players_[player];
        pl->notify("Select an option:");
        for (int i = 0; i < size; ++i) {
          auto opt = read_u32();
          std::string s;
          if (opt > 10000) {
            CardCode code = opt >> 4;
            s = c_get_card(code).strings_[opt & 0xf];
          } else {
            s = get_system_string(opt);
          }
          std::string option = std::to_string(i + 1);
          options_.push_back(option);
          pl->notify(option + ": " + s);
        }
      } else {
        for (int i = 0; i < size; ++i) {
          dp_ += 4;
          options_.push_back(std::to_string(i + 1));
        }
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        if (verbose_) {
          players_[to_play_]->notify("You selected option " + options_[idx] +
                                     ".");
          players_[1 - to_play_]->notify(players_[to_play_]->nickname_ +
                                         " selected option " + options_[idx] +
                                         ".");
        }

        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_SELECT_IDLECMD) {
      int32_t player = read_u8();
      auto summonable_ = read_cardlist_spec(player);
      auto spsummon_ = read_cardlist_spec(player);
      auto repos_ = read_cardlist_spec(player);
      auto idle_mset_ = read_cardlist_spec(player);
      auto idle_set_ = read_cardlist_spec(player);
      auto idle_activate_ = read_cardlist_spec(player, true);
      bool to_bp_ = read_u8();
      bool to_ep_ = read_u8();
      read_u8(); // can_shuffle

      int offset = 0;

      auto pl = players_[player];
      if (verbose_) {
        pl->notify("Select a card and action to perform.");
      }
      for (const auto &[code, spec, data] : summonable_) {
        std::string option = "s " + spec;
        options_.push_back(option);
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          pl->notify(option + ": Summon " + name +
                     " in face-up attack position.");
        }
      }
      offset += summonable_.size();
      int spsummon_offset = offset;
      for (const auto &[code, spec, data] : spsummon_) {
        std::string option = "c " + spec;
        options_.push_back(option);
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          pl->notify(option + ": Special summon " + name + ".");
        }
      }
      offset += spsummon_.size();
      int repos_offset = offset;
      for (const auto &[code, spec, data] : repos_) {
        std::string option = "r " + spec;
        options_.push_back(option);
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          pl->notify(option + ": Reposition " + name + ".");
        }
      }
      offset += repos_.size();
      int mset_offset = offset;
      for (const auto &[code, spec, data] : idle_mset_) {
        std::string option = "m " + spec;
        options_.push_back(option);
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          pl->notify(option + ": Summon " + name +
                     " in face-down defense position.");
        }
      }
      offset += idle_mset_.size();
      int set_offset = offset;
      for (const auto &[code, spec, data] : idle_set_) {
        std::string option = "t " + spec;
        options_.push_back(option);
        if (verbose_) {
          const auto &name = c_get_card(code).name_;
          pl->notify(option + ": Set " + name + ".");
        }
      }
      offset += idle_set_.size();
      int activate_offset = offset;
      ankerl::unordered_dense::map<std::string, int> idle_activate_count;
      for (const auto &[code, spec, data] : idle_activate_) {
        idle_activate_count[spec] += 1;
      }
      ankerl::unordered_dense::map<std::string, int> activate_count;
      for (const auto &[code, spec, data] : idle_activate_) {
        // TODO: use effect description to indicate which effect to activate
        std::string option = "v " + spec;
        int count = idle_activate_count[spec];
        activate_count[spec]++;
        if (count > 1) {
          option.push_back('a' + activate_count[spec] - 1);
        }
        options_.push_back(option);
        if (verbose_) {
          pl->notify(option + ": " +
                     c_get_card(code).get_effect_description(data));
        }
      }

      if (to_bp_) {
        std::string cmd = "b";
        options_.push_back(cmd);
        if (verbose_) {
          pl->notify(cmd + ": Enter the battle phase.");
        }
      }
      if (to_ep_) {
        if (!to_bp_) {
          std::string cmd = "e";
          options_.push_back(cmd);
          if (verbose_) {
            pl->notify(cmd + ": End phase.");
          }
        }
      }

      to_play_ = player;
      callback_ = [this, spsummon_offset, repos_offset, mset_offset, set_offset,
                   activate_offset](int idx) {
        const auto &option = options_[idx];
        char cmd = option[0];
        if (cmd == 'b') {
          YGO_SetResponsei(pduel_, 6);
        } else if (cmd == 'e') {
          YGO_SetResponsei(pduel_, 7);
        } else {
          auto spec = option.substr(2);
          if (cmd == 's') {
            uint32_t idx_ = idx;
            YGO_SetResponsei(pduel_, idx_ << 16);
          } else if (cmd == 'c') {
            uint32_t idx_ = idx - spsummon_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 1);
          } else if (cmd == 'r') {
            uint32_t idx_ = idx - repos_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 2);
          } else if (cmd == 'm') {
            uint32_t idx_ = idx - mset_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 3);
          } else if (cmd == 't') {
            uint32_t idx_ = idx - set_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 4);
          } else if (cmd == 'v') {
            uint32_t idx_ = idx - activate_offset;
            YGO_SetResponsei(pduel_, (idx_ << 16) + 5);
          } else {
            throw std::runtime_error("Invalid option: " + option);
          }
        }
      };
    } else if (msg_ == MSG_SELECT_PLACE) {
      auto player = read_u8();
      auto count = read_u8();
      if (count == 0) {
        count = 1;
      }
      auto flag = read_u32();
      options_ = flag_to_usable_cardspecs(flag);
      if (verbose_) {
        std::string specs_str = options_[0];
        for (int i = 1; i < options_.size(); ++i) {
          specs_str += ", " + options_[i];
        }
        if (count == 1) {
          players_[player]->notify("Select place for card, one of " +
                                   specs_str + ".");
        } else {
          players_[player]->notify("Select " + std::to_string(count) +
                                   " places for card, from " + specs_str + ".");
        }
      }
      to_play_ = player;
      callback_ = [this, player](int idx) {
        int y = player + 1;
        std::string spec = options_[idx];
        auto plr = player;
        if (spec[0] == 'o') {
          plr = 1 - player;
          spec = spec.substr(1);
        }
        auto [loc, seq, pos] = spec_to_ls(spec);
        resp_buf_[0] = plr;
        resp_buf_[1] = loc;
        resp_buf_[2] = seq;
        YGO_SetResponseb(pduel_, resp_buf_);
      };
    } else if (msg_ == MSG_SELECT_DISFIELD) {
      auto player = read_u8();
      auto count = read_u8();
      if (count == 0) {
        count = 1;
      }
      auto flag = read_u32();
      options_ = flag_to_usable_cardspecs(flag);
      if (verbose_) {
        std::string specs_str = options_[0];
        for (int i = 1; i < options_.size(); ++i) {
          specs_str += ", " + options_[i];
        }
        if (count == 1) {
          players_[player]->notify("Select place for card, one of " +
                                   specs_str + ".");
        } else {
          throw std::runtime_error("Select disfield count " +
                                   std::to_string(count) + " not implemented");
        }
      }
      to_play_ = player;
      callback_ = [this, player](int idx) {
        int y = player + 1;
        std::string spec = options_[idx];
        auto plr = player;
        if (spec[0] == 'o') {
          plr = 1 - player;
          spec = spec.substr(1);
        }
        auto [loc, seq, pos] = spec_to_ls(spec);
        resp_buf_[0] = plr;
        resp_buf_[1] = loc;
        resp_buf_[2] = seq;
        YGO_SetResponseb(pduel_, resp_buf_);
      };
    } else if (msg_ == MSG_SELECT_COUNTER) {
      auto player = read_u8();
      auto counter_type = read_u16();
      int counter_count = read_u16();
      int count = read_u8();
      if (count > 2) {
        throw std::runtime_error("Select counter count " +
                                 std::to_string(count) + " not implemented");
      }
      auto pl = players_[player];
      if (verbose_) {
        pl->notify(fmt::format("Type new {} for {} card(s), separated by spaces.", "UNKNOWN_COUNTER", count));
      }
      std::vector<int> counters;
      counters.reserve(count);
      for (int i = 0; i < count; ++i) {
        auto code = read_u32();
        auto controller = read_u8();
        auto loc = read_u8();
        auto seq = read_u8();
        auto counter = read_u16();
        counters.push_back(counter & 0xffff);

        if (verbose_) {
          pl->notify(c_get_card(code).name_ + ": " + std::to_string(counter));
        }
        // auto spec = ls_to_spec(loc, seq, 0, controller != player);
        // options_.push_back(spec);
      }
      // TODO: implement action
      n_counters_ = count;
      uint16_t resp1 = static_cast<uint16_t>(std::min(counter_count, counters[0]));
      memcpy(resp_buf_, &resp1, 2);
      counter_count -= counters[0];
      if (count == 2) {
        uint16_t resp2 = 0;
        if (counter_count > 0) {
          resp2 = static_cast<uint16_t>(counter_count);
        }
        memcpy(resp_buf_ + 2, &resp2, 2);
      }
      YGO_SetResponseb(pduel_, resp_buf_);
    } else if (msg_ == MSG_ANNOUNCE_NUMBER) {
      auto player = read_u8();
      int count = read_u8();
      std::vector<int> numbers;
      for (int i = 0; i < count; ++i) {
        int number = read_u32();
        if (number <= 0 || number > 12) {
          throw std::runtime_error("Number " + std::to_string(number) +
                                   " not implemented for announce number");
        }
        numbers.push_back(number);
        options_.push_back(std::string(1, '0' + number));
      }
      if (verbose_) {
        auto pl = players_[player];
        std::string str = "Select a number, one of: [";
        for (int i = 0; i < count; ++i) {
          str += std::to_string(numbers[i]);
          if (i < count - 1) {
            str += ", ";
          }
        }
        str += "]";
        pl->notify(str);
      }
      to_play_ = player;
      callback_ = [this](int idx) {
        YGO_SetResponsei(pduel_, idx);
      };
    } else if (msg_ == MSG_ANNOUNCE_ATTRIB) {
      auto player = read_u8();
      int count = read_u8();
      auto flag = read_u32();

      int n_attrs = 7;

      std::vector<uint8_t> attrs;
      for (int i = 0; i < n_attrs; i++) {
        if (flag & (1 << i)) {
          attrs.push_back(i + 1);
        }
      }

      if (count != 1) {
        throw std::runtime_error("Announce attrib count " +
                                 std::to_string(count) + " not implemented");
      }

      if (verbose_) {
        auto pl = players_[player];
        pl->notify("Select " + std::to_string(count) +
                   " attributes separated by spaces:");
        for (int i = 0; i < attrs.size(); i++) {
          pl->notify(std::to_string(attrs[i]) + ": " +
                     attribute_to_string(1 << (attrs[i] - 1)));
        }
      }

      auto combs = combinations(attrs.size(), count);
      for (const auto &comb : combs) {
        std::string option = "";
        for (int j = 0; j < count; ++j) {
          option += std::to_string(attrs[comb[j]]);
          if (j < count - 1) {
            option += " ";
          }
        }
        options_.push_back(option);
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        const auto &option = options_[idx];
        uint32_t resp = 0;
        int i = 0;
        while (i < option.size()) {
          resp |= 1 << (option[i] - '1');
          i += 2;
        }
        YGO_SetResponsei(pduel_, resp);
      };

    } else if (msg_ == MSG_SELECT_POSITION) {
      // TODO: add card as feature
      auto player = read_u8();
      auto code = read_u32();
      auto valid_pos = read_u8();

      if (verbose_) {
        auto pl = players_[player];
        auto card = c_get_card(code);
        pl->notify("Select position for " + card.name_ + ":");
      }

      std::vector<uint8_t> positions;
      int i = 1;
      for (auto pos : {POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK,
                       POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE}) {
        if (valid_pos & pos) {
          positions.push_back(pos);
          options_.push_back(std::to_string(i));
          if (verbose_) {
            auto pl = players_[player];
            pl->notify(fmt::format("{}: {}", i, position_to_string(pos)));
          }
        }
        i++;
      }

      to_play_ = player;
      callback_ = [this](int idx) {
        uint8_t pos = options_[idx][0] - '1';
        YGO_SetResponsei(pduel_, 1 << pos);
      };
    } else {
      show_deck(0);
      show_deck(1);
      show_buffer();
      throw std::runtime_error(
        fmt::format("Unknown message {}, length {}, dp {}",
        msg_to_string(msg_), dl_, dp_));
    }
  }

  void _damage(uint8_t player, uint32_t amount) {
    lp_[player] -= amount;
    if (verbose_) {
      auto lp = players_[player];
      lp->notify(fmt::format("Your lp decreased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp decreased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _recover(uint8_t player, uint32_t amount) {
    lp_[player] += amount;
    if (verbose_) {
      auto lp = players_[player];
      lp->notify(fmt::format("Your lp increased by {}, now {}", amount, lp_[player]));
      players_[1 - player]->notify(fmt::format("{}'s lp increased by {}, now {}",
                                   lp->nickname_, amount, lp_[player]));
    }
  }

  void _duel_end(uint8_t player, uint8_t reason) {
    winner_ = player;
    win_reason_ = reason;
    if (async_reset_) {
      n_lives_--;
    } else {
      YGO_EndDuel(pduel_);
    }

    duel_started_ = false;
  }
};

using YGOProEnvPool = AsyncEnvPool<YGOProEnv>;

} // namespace ygopro0

#endif // YGOENV_YGOPRO0_YGOPRO_H_
