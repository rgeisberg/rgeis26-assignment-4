install:
	python3.12 -m venv venv
	. venv/bin/activate && pip3 install -r requirements.txt

run:
	. venv/bin/activate && flask run --host=0.0.0.0 --port=3000
