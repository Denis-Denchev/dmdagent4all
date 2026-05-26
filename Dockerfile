FROM node:22-alpine AS dashboard

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    XDG_DATA_HOME=/data \
    DMDCORE_STATIC_DIR=/app/frontend/dist \
    DMDCORE_WORKSPACE=/workspace

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY scripts/docker-entrypoint.sh /usr/local/bin/dmdcore-docker-entrypoint
COPY --from=dashboard /app/frontend/dist ./frontend/dist

RUN sed -i 's/\r$//' /usr/local/bin/dmdcore-docker-entrypoint \
    && chmod +x /usr/local/bin/dmdcore-docker-entrypoint \
    && pip install --no-cache-dir .

EXPOSE 8765

ENTRYPOINT ["dmdcore-docker-entrypoint"]
CMD ["serve"]
