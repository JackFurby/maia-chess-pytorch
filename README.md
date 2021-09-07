# maia-chess-pytorch
A maia chess engine but implimented in PyTorch



### Dataset

1. download pgn-extract, unzip and run `make` in the directory
2. `export PATH=$PATH:/home/jack/Documents/programming/from_papers/aligning_superhuman_ai_with_human_behavior__chess/pgn-extract`
3. download trainingdata-tool, unzip and follow the installation guide from http://phoenix.yizimg.com/kennyfrc/trainingdata-tool
4. `export PATH=$PATH:/home/jack/Documents/programming/from_papers/aligning_superhuman_ai_with_human_behavior__chess/trainingdata-tool`
5. `export PATH=$PATH:/home/jack/Documents/programming/from_papers/aligning_superhuman_ai_with_human_behavior__chess/lc0/build/release`





### Useful links

https://chess.stackexchange.com/questions/28321/is-there-python-code-to-use-the-leela-neural-network
https://github.com/so-much-meta/lczero_tools


### convert model weights to Tensorflow model

`python move_prediction/maia_chess_backend/maia/net_to_model.py move_prediction/model_files/1100/final_1100-40.pb.gz --cfg move_prediction/maia_config.yaml` - weights to tensorflow checkpoint


### pb.gz to txt.gz

`python move_prediction/maia_chess_backend/maia/net.py -i ./maia_weights/maia-1900.pb.gz -o ./maia_weights/maia-1900.txt.gz`
