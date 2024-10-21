# install dependencies
echo "Installing dependencies..."
python3 -m pip install --no-cache-dir --upgrade -r requirements.txt
python3 -m pip install numpy==1.26.2
python3 -m pip install urllib3==1.26.6

echo "Installing flash-attention..."
cd ../flash-attention
python3 -m pip install wheel==0.41.3
python3 setup.py install

echo "Installation completed!"

cd ../LLaVA_KD/bitsandbytes
pip install -e .
cd ..
pip install ptflops
pip install deepspeed==0.14.4