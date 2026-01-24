curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get update -y
sudo apt-get install -y python3-venv

python3 -m venv llm-env
source llm-env/bin/activate
python3 -m pip install --upgrade pip
pip install -r llm-requirements.txt
pip install -U scikit-learn
python -c "import torch; print(torch.cuda.is_available())" # This should return True

# Flash attention installation
cd ..
pip install wheel
pip install ninja packaging wheel cmake
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e . --no-build-isolation
