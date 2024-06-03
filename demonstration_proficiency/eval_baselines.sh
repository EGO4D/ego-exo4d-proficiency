echo "RESULTS: Random"
python eval_baselines.py ./configs/egoexo_omni_arxivb.yaml -b random
#
echo "RESULTS: Positive/"
python eval_baselines.py ./configs/egoexo_omni_arxivb.yaml -b positive
#
echo "RESULTS: Negative/"
python eval_baselines.py ./configs/egoexo_omni_arxivb.yaml -b negative
## 709252893
