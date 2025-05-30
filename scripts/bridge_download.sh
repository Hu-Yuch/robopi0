models=(
    IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge
)
for model in ${models[@]};
do
  echo downloading ${model}...
  huggingface-cli download --resume-download --local-dir-use-symlinks False ${model} \
  --local-dir ../pretrained/$(basename ${model})
done