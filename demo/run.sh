export CUDA_VISIBLE_DEVICES=1
python main.py \
    --model_path /workspace/data/SNU_Eng_Retrieval/results/BAAI__bge-m3/v2.14_bge-m3-merged \
    --research_corpus_path /workspace/data/SNU_Eng_Retrieval/translation_data/research_translation_final.csv \
    --project_corpus_path /workspace/data/SNU_Eng_Retrieval/translation_data/project_translation_final.csv \
    --index_path /workspace/data/SNU_Eng_Retrieval/demo/ver1 \
    --device cuda:1