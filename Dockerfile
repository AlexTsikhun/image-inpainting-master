FROM python:3.12-slim

# install dependencies for cv2
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev --frozen

COPY . .

# Вказуємо команду запуску
CMD ["uv", "run", "python", "application.py"]
