# Warhammer 40K Tabletop AI

Reinforcement learning agent for Warhammer 40K tabletop combat.
Uses a custom game simulation engine with a PettingZoo/Gymnasium RL environment,
trained via MaskablePPO self-play to produce a competitive AI opponent.

## Architecture

```
Warhammer-AI/
├── wh_types.py          # Core datatypes: Unit, Weapon, GameState, Terrain, etc.
├── unit_data.py         # Faction unit definitions (Tau, Ultramarines)
├── combat.py            # Shooting resolution (hit -> wound -> save -> damage)
├── core.py              # Melee resolution + target queries
├── los.py               # Shapely-based 2.5D line of sight
├── movement.py          # Continuous-coordinate movement + collision + charges
├── game_engine.py       # Phase state machine, turn management, action execution
├── rl_enviornment.py    # Matplotlib battlefield visualization
│
├── env/
│   ├── wh40k_env.py     # PettingZoo AEC + Gymnasium single-agent environments
│   ├── action_codec.py  # MultiDiscrete action encoding/decoding + masks
│   ├── obs_builder.py   # Observation space encoding
│   └── reward.py        # Layered reward shaping
│
├── training/
│   ├── train.py         # MaskablePPO training entry point
│   ├── self_play.py     # Self-play callback + opponent pool
│   └── tournament.py    # Round-robin tournament with ELO ratings
│
├── bots/
│   ├── random_bot.py    # Random legal action baseline
│   └── heuristic_bot.py # Rule-based opponent (focus fire, advance to objectives)
│
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

### Phase 1: Train against random bot
```bash
python -m training.train --opponent random --timesteps 200000
```

### Phase 2: Train against heuristic bot
```bash
python -m training.train --opponent heuristic --timesteps 500000
```

### Phase 3: Self-play
```bash
python -m training.train --opponent self-play --timesteps 1000000
```

### Resume from checkpoint
```bash
python -m training.train --opponent self-play --timesteps 500000 --resume models/wh40k_agent_final
```

## Tournament

Run a round-robin tournament between saved checkpoints:
```bash
python -m training.tournament --model-dir models/opponents --num-games 20 --include-random
```

## Game Rules (Simplified 40K)

Each player's turn runs through four phases:
1. **Movement** -- move units up to their M characteristic, checking collisions
2. **Shooting** -- fire ranged weapons at enemies in range with line of sight
3. **Charge** -- roll 2d6 to charge into melee engagement range
4. **Fight** -- melee attacks against engaged enemies

The game runs for 5 turns. Victory points are scored by controlling objectives.
The player with more VP at the end wins. If all of one player's units are
destroyed, the other player wins immediately.

## Adding Factions

Add new units in `unit_data.py` following the existing pattern:
1. Define weapons as `Weapon` instances
2. Create unit factory functions returning `Unit`
3. Write an army builder function that places units on the board
