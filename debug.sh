export CUDA_VISIBLE_DEVICES=0
mkdir model/YoutubeDebug
python3 AgentForEdgeGAN.py /ldev/wsx/tmp/netemb/github/dataset/generated_data/eco_youtube2.txt.labeled.reindex YoutubeDebug
python3 Predictor.py /ldev/wsx/tmp/netemb/github/dataset/generated_data/eco_youtube2.txt.labeled.reindex YoutubeDebug res/youtubeDebug.txt