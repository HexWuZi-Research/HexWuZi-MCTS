# HexWuZi-MCST

## Intro

The basic demo of using Monte carlo tree search to play Hexgonal WuZi(Five in a Row).

## Dependency

We use numba to accelerate, so you should install it.

```shell
pip install -r requirements.txt
```

## AI self-play

Run this command in a terminal,

```shell
python mcts.py
```

## Play in command line with AI

Run this command in a terminal,

```shell
python cmdgame.py
```

Just input position in form of "5,5" or "4 6".

## Developing Log

### 21/11/2022

1. Optimize board check function.
2. Complete the basic backbone of MCTS on HexWuZi game.
3. Add only explore adajency policy in order to let MCTS to get an at least acceptable performance.

### 23/11/2022

1. Accelerate HexWuZiState with numba.experimential jitclass.
2. Optimize current expand function in MCTS.
3. Add player interacting game in cmd.

### 25/11/2022

1. Utilize the searched state node in the tree.