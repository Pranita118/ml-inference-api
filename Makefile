.PHONY: install train test run docker-build docker-run
install:      ; pip install -r requirements-dev.txt
train:        ; python train.py
test:         ; pytest -v
run:          ; uvicorn app.main:app --reload
docker-build: ; docker build -t ml-inference-api .
docker-run:   ; docker run --rm -p 8000:8000 -e API_KEY=$${API_KEY} ml-inference-api
