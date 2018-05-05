
mkdir model/Blog
mkdir model/Youtube
mkdir model/FLICKR

export CUDA_VISIBLE_DEVICES=09
python3 AgentForEdgeGAN.py /ldev/wsx/tmp/netemb/github/generated_data/eco_blogCatalog.txt.labeled.reindex Blog
python3 AgentForEdgeGAN.py /ldev/wsx/tmp/netemb/github/generated_data/eco_youtube.txt.labeled.reindex Youtube
python3 AgentForEdgeGAN.py /ldev/wsx/tmp/netemb/github/generated_data/eco_nips.txt.labeled.reindex NIPS
python3 AgentForEdgeGAN.py /ldev/wsx/tmp/netemb/github/generated_data/eco_flickr.txt.labeled.reindex FLICKR