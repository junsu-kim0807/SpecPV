环境记录： zhendong_sparsesd python=3.11

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 pytorch 2.7.1 cu118
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

pip install -e . [dev]

问题检查
ruff check . 
ruff check . --fix
自动缩进
black .
格式化import
isort . 