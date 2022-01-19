autoflake8 --in-place --remove-unused-variables output.py
autopep8 --in-place --aggressive --aggressive -a -a output.py
isort output.py