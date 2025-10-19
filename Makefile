IMAGE ?= docker.io/your-dockerhub-username/db-structure-analysis
TAG ?= latest

.PHONY: build-amd64 push-amd64 run compose-up compose-down login

build-amd64:
	docker buildx create --use --name builder 2>/dev/null || true
	docker buildx build \
		--platform linux/amd64 \
		--tag $(IMAGE):$(TAG) \
		--file Dockerfile \
		--load \
		.

push-amd64:
	docker buildx create --use --name builder 2>/dev/null || true
	docker buildx build \
		--platform linux/amd64 \
		--tag $(IMAGE):$(TAG) \
		--file Dockerfile \
		--push \
		.

run:
	OPENAI_API_KEY=$${OPENAI_API_KEY} OPENAI_BASE_URL=$${OPENAI_BASE_URL} OPENAI_MODEL=$${OPENAI_MODEL} \
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down

login:
	@echo "docker login docker.io"
