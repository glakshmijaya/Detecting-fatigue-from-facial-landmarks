FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN find /etc/apt/ -name "*.sources" -type f -print0 | \
    xargs -0 sed -i 's|http://deb.debian.org|https://deb.debian.org|g' && \
    find /etc/apt/ -name "*.sources" -type f -print0 | \
    xargs -0 sed -i 's|http://security.debian.org|https://security.debian.org|g'

# Install system dependencies required by OpenCV & Mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    curl \
    gnupg \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir streamlit

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "fatigue_dashboard1.py", "--server.port=8501", "--server.address=0.0.0.0"]
