# install python and pip, don't modify this, modify install_python_package.sh
apt-get update
apt-get install -y python-dev python3-dev

# install pip
cd /tmp && wget https://bootstrap.pypa.io/get-pip.py
python2 get-pip.py
python3 get-pip.py

# santiy check
python2 --version
python3 --version
pip2 --version
pip3 --version
