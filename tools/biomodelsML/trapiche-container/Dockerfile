FROM jhoanmanuelms/trapiche-inference@sha256:45c9a6ff29791270c5ea39671a634ac3e4397f149f5488b8825c3a6d48937fb6

LABEL org.opencontainers.image.title="Trapiche text classification for Galaxy" \
      org.opencontainers.image.description="CPU text-to-biome classification runtime from BioAIrepo study S-BAIR13" \
      org.opencontainers.image.version="1.1.4-s-bair13" \
      org.opencontainers.image.source="https://www.ebi.ac.uk/biostudies/bioairepo/studies/S-BAIR13" \
      org.opencontainers.image.documentation="https://doi.org/10.6019/S-BAIR13" \
      org.opencontainers.image.base.digest="sha256:45c9a6ff29791270c5ea39671a634ac3e4397f149f5488b8825c3a6d48937fb6"

# Match verification to the files deposited in BioStudies S-BAIR13 while
# retaining strict size and checksum checks for every runtime asset.
COPY patch_model_manifest.py /tmp/patch_model_manifest.py
RUN python3 /tmp/patch_model_manifest.py && rm /tmp/patch_model_manifest.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TRAPICHE_TEXT_DEVICE=cpu \
    TRAPICHE_TEXT_SPLIT_SENTENCES=false \
    TRAPICHE_TEXT_BATCH_SIZE=1

ENTRYPOINT []
CMD ["trapiche-galaxy", "--help"]
