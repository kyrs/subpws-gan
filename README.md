# subpws-gan
The current codebase is built upon the WSGAN repo https://github.com/benbo/WSGAN-paper.
<br>
We have provided the LF-file for GTSRB. Other label function can be downloaded from the original repo. 
## Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir outputs
# embedding generation
bash embedding_generation.sh
# Running model
bash run_gtsrb.sh
```
