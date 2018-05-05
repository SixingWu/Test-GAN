rm -r -f model/Blog
rm -r -f model/Youtube
rm -r -f model/FLICKR
mkdir model/Blog
mkdir model/Youtube
mkdir model/FLICKR
export CUDA_VISIBLE_DEVICES=1
python3 AgentForEdgeGAN.py /ldev/wsx/tmp/netemb/github/dataset/generated_data/eco_blogCatalog3.txt.labeled.reindex Blog
python3 AgentForEdgeGAN.py /ldev/wsx/tmp/netemb/github/dataset/generated_data/eco_youtube2.txt.labeled.reindex Youtube
python3 AgentForEdgeGAN.py /ldev/wsx/tmp/netemb/github/dataset/generated_data/eco_flickr.txt.labeled.reindex FLICKR