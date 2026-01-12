## Environment Setup
```bash
conda create --name da python=3.11 -y
conda activate da
```

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install diffusers
```

Check if GPU is available
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
