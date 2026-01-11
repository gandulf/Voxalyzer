## Setup

Before you can run the Voxalyzer [Music2Emotion](https://github.com/AMAAI-Lab/Music2Emotion) has to be setup by installing it's requirements.

```bash
cd Music2Emotion
pip install -r requirements.txt
```
#### Enable GPU Support (Optional)
If you have a good GPU you can switch from CPU Mode to GPU mode by installing a Cuda Enabled Version of Torch
```bash
pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

Back in the base directory you can run `music2emotion.py` with the directory to analyze as argument.
```bash
python.exe .\music2emotion.py "C:\Users\paulsmith\Music\" 
```

> [!Note]
> Depending on your hardware a single ~3min song should take about 1-5seconds to analyze.