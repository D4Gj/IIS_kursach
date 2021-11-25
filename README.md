# Start

## Requirements

```ps1
python -m venv venv
.\venv\scripts\activate

pip install -r requirements.txt
```

## Start

Создать файл `.env`

```text
FLASK_ENV=development
FLASK_APP=main.py
```

```ps1
.\venv\scripts\activate
python -m flask run -p 9000
```
