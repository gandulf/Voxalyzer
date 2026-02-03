
To install dependencies under windows
```bash
pip install .[dev] --find-links ./wheels
```
To build under windows:
```bash
python -m nuitka --jobs=16 voxalyzer.py --product-version=0.1.0.0 --file-version=0.1.0.0
```

To use docker image to analyze local directory
```bash
docker run --gpus all -v C:/Users/gandu/Music/Test:/music voxalyzer /music --force
```

To run webserver accepting calls under port 8000 /analyze
```bash
docker run --gpus all -p 8000:8000 voxalyzer
```