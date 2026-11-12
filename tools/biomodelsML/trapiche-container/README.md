# Trapiche text-classification container

This image specializes the released Linux AMD64 CPU runtime documented by
BioAIrepo study [S-BAIR13](https://www.ebi.ac.uk/biostudies/bioairepo/studies/S-BAIR13)
for offline Galaxy text classification. The upstream runtime is pinned by image
digest and contains Trapiche 1.1.4 from source revision
`985f1cc423673fd88dbf1813e64934d4315429af`.

Build from the Galaxy repository root:

```bash
docker build --platform linux/amd64 \
  -t trapiche-text-s-bair13:1.1.4-1 \
  tools/biomodelsML/trapiche-container
```

The model weights remain Galaxy inputs and are not copied into the image. The
tool stages the six executable text-classifier files and six vector-support
files listed by S-BAIR13, after which `trapiche-galaxy` verifies their pinned
checksums. The text model-card README is documentation and is not required.
