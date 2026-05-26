#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_REPO="${MODEL_REPO:-jAEhEEkIM/SNUResearcherMatching-model}"
DATA_REPO="${DATA_REPO:-jAEhEEkIM/SNUResearcherMatching-demo-data}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

download() {
  local repo_type="$1"
  local repo_id="$2"
  local remote_path="$3"
  local local_path="$4"
  local base_url

  if [[ "${repo_type}" == "dataset" ]]; then
    base_url="${HF_ENDPOINT}/datasets/${repo_id}/resolve/main"
  else
    base_url="${HF_ENDPOINT}/${repo_id}/resolve/main"
  fi

  mkdir -p "$(dirname "${local_path}")"
  if [[ -s "${local_path}" ]]; then
    echo "Already exists: ${local_path}"
    return
  fi

  echo "Downloading ${repo_id}/${remote_path}"
  local partial_path="${local_path}.part"
  curl --fail --location --continue-at - \
    "${base_url}/${remote_path}" \
    --output "${partial_path}"
  mv "${partial_path}" "${local_path}"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command curl

mkdir -p \
  "${REPO_ROOT}/trained_model/1_Pooling" \
  "${REPO_ROOT}/trained_model/2_Normalize" \
  "${REPO_ROOT}/data/demo/ver1" \
  "${REPO_ROOT}/data/new_researchers"

download model "${MODEL_REPO}" ".gitattributes" "${REPO_ROOT}/trained_model/.gitattributes"
download model "${MODEL_REPO}" "1_Pooling/config.json" "${REPO_ROOT}/trained_model/1_Pooling/config.json"
download model "${MODEL_REPO}" "config%20copy.json" "${REPO_ROOT}/trained_model/config copy.json"
download model "${MODEL_REPO}" "config.json" "${REPO_ROOT}/trained_model/config.json"
download model "${MODEL_REPO}" "config_sentence_transformers.json" "${REPO_ROOT}/trained_model/config_sentence_transformers.json"
download model "${MODEL_REPO}" "model.safetensors" "${REPO_ROOT}/trained_model/model.safetensors"
download model "${MODEL_REPO}" "modules.json" "${REPO_ROOT}/trained_model/modules.json"
download model "${MODEL_REPO}" "sentence_bert_config.json" "${REPO_ROOT}/trained_model/sentence_bert_config.json"
download model "${MODEL_REPO}" "sentencepiece.bpe.model" "${REPO_ROOT}/trained_model/sentencepiece.bpe.model"
download model "${MODEL_REPO}" "special_tokens_map.json" "${REPO_ROOT}/trained_model/special_tokens_map.json"
download model "${MODEL_REPO}" "tokenizer.json" "${REPO_ROOT}/trained_model/tokenizer.json"
download model "${MODEL_REPO}" "tokenizer_config.json" "${REPO_ROOT}/trained_model/tokenizer_config.json"

download dataset "${DATA_REPO}" "project_translation_final.csv" "${REPO_ROOT}/data/project_translation_final.csv"
download dataset "${DATA_REPO}" "research_translation_final.csv" "${REPO_ROOT}/data/research_translation_final.csv"
download dataset "${DATA_REPO}" "demo/ver1/full.faiss" "${REPO_ROOT}/data/demo/ver1/full.faiss"
download dataset "${DATA_REPO}" "demo/ver1/subset.faiss" "${REPO_ROOT}/data/demo/ver1/subset.faiss"
download dataset "${DATA_REPO}" "new_researchers/project_new_prof_10.txt" "${REPO_ROOT}/data/new_researchers/project_new_prof_10.txt"
download dataset "${DATA_REPO}" "new_researchers/research_new_prof_10.txt" "${REPO_ROOT}/data/new_researchers/research_new_prof_10.txt"

echo "Artifacts are ready under:"
echo "  ${REPO_ROOT}/trained_model"
echo "  ${REPO_ROOT}/data"
