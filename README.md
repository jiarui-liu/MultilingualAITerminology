**This is the repository for the preprint [Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](https://arxiv.org/abs/2412.18367).**

**Dataset link: https://huggingface.co/datasets/Jerry999/multilingual-terminology**


## Website Demo

### Environment setup:

```
conda create -n gist python=3.10
conda activate gist
pip install -r requirements.txt
```

### Download the Dataset:

```
python3 download_dataset.py
```

### Run the Demo

Please refer to [server/README.md](server/README.md) for detailed instructions on setting up and launching the demo.

## Citation

If you find our work useful, please cite:

```bibtex
@article{liu2024towards,
  title={Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset},
  author={Liu, Jiarui and Ouzzani, Iman and Li, Wenkai and Zhang, Lechen and Ou, Tianyue and Bouamor, Houda and Jin, Zhijing and Diab, Mona},
  journal={arXiv preprint arXiv:2412.18367},
  year={2024}
}
```
