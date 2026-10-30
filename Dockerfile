# Chạy trên CPU để chia sẻ được cho máy không có GPU NVIDIA.
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TORCH_HOME=/app/.cache/torch \
    TMPDIR=/app/.cache/tmp

WORKDIR /app

# OpenMP cần cho các thư viện tính toán như scikit-learn/PyTorch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tái sử dụng layer khi chỉ thay đổi code.
# Cài bản Torch CPU riêng, tránh tải thêm các thư viện CUDA lớn.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY phobert_features.py test_phobert_features.py ./
COPY phobert_pipeline/ ./phobert_pipeline/
COPY data/ ./data/
RUN mkdir -p /app/.cache/tmp /app/outputs

ENTRYPOINT ["python", "-B", "phobert_features.py"]
CMD []
