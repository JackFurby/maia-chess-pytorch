import chess
import chess.engine
import os

import subprocess

lc0_path = "./../../../aligning_superhuman_ai_with_human_behavior__chess/lc0/build/release/lo0"
lc0_path = "/home/jack/Documents/programming/from_papers/aligning_superhuman_ai_with_human_behavior__chess/lc0/build/release/lc0"
weightsPath = os.path.abspath("./maia-1100.pb.gz")
threads = 1
backend = "blas"
backend_opts = ""
temperature = 0
temp_decay = 0
noise = False
verbose = False
extra_flags = None

#engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")
#engine = chess.engine.SimpleEngine.popen_uci([lc0_path, f'--weights={weightsPath}', f'--threads={threads}', f'--backend={backend}', f'--backend-opts={backend_opts}', f'--temperature={temperature}', f'--tempdecay-moves={temp_decay}'] + (['--noise'] if noise else []) + ([f'--noise-epsilon={noise}'] if isinstance(noise, float) else []) + (['--verbose-move-stats'] if verbose else []) + (extra_flags if extra_flags is not None else []), stderr=subprocess.DEVNULL)
engine = chess.engine.SimpleEngine.popen_uci([lc0_path, f'--weights={weightsPath}', f'--backend-opts={backend_opts}'])

board = chess.Board()
info = engine.analyse(board, chess.engine.Limit(time=0.1))
print("Score:", info["score"])
# Score: PovScore(Cp(+20), WHITE)

board = chess.Board("r1bqkbnr/p1pp1ppp/1pn5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 4")
info = engine.analyse(board, chess.engine.Limit(depth=20))
print("Score:", info["score"])
# Score: PovScore(Mate(+1), WHITE)

engine.quit()
