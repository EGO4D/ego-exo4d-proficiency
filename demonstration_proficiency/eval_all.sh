echo "RESULTS: Exo"
python eval.py configs/egoexo_omni_arxivb.yaml ckpt/egoexo_omni_arxiv_9G8OM8K1DEWX4CDZK520UUJBHSISO8/epoch_010.pth.tar
#
echo "RESULTS: Ego"
python eval.py configs/egoexo_omni_arxivb_ego.yaml ckpt/egoexo_omni_arxiv_M6GV6N8IYCHOROGUC4KJC9SRYZFV2B/epoch_010.pth.tar
#
echo "RESULTS: EgoExo"
python eval.py configs/egoexo_omni_arxivb_egoexo.yaml ckpt/egoexo_omni_arxiv_WO4S5WPI2HSRWK4TNV73AGMYP3JXEJ/epoch_005.pth.tar
## 709252893
