python3 -m venv env_exist2021
source env_exist2021/bin/activate
git clone https://github.com/franciscorodriguez92/exist2021.git
cd exist2021/src
pip3 install -r requirements.txt
mv get_data.py ./exist2021/exist2021/src/
python3 get_data.py
python3 train.py --sample
python3 generate_submissions.py --sample
