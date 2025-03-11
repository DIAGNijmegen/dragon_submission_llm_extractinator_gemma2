FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y curl wget

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /opt/tiktoken_cache /input /output && chown user:user /opt/app /opt/tiktoken_cache /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

# Install the requirements
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user -r requirements.txt

# Move the model weights to the right folder
# COPY --chown=user models /opt/ml/model/
ENV OLLAMA_MODELS=/opt/ml/model/

# Download the model, tokenizer and metrics
# COPY --chown=user:user download_model.py /opt/app/
# RUN python download_model.py --model_name distilbert-base-multilingual-cased
# Adapt to the model you want to download, e.g.:
# RUN python download_model.py --model_name joeranbosma/dragon-roberta-base-mixed-domain
COPY --chown=user:user download_metrics.py /opt/app/
RUN python download_metrics.py
COPY --chown=user:user llm_extractinator /opt/app/llm_extractinator

# Set the environment variables
ENV TRANSFORMERS_OFFLINE=1
ENV HF_EVALUATE_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Copy the algorithm code
COPY --chown=user:user process.py /opt/app/

# Cache tiktoken
# COPY --chown=user:user tiktoken/6bac041639f96ec8dfd77f9156d953f52fa49279 /opt/tiktoken_cache/
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN pip install --upgrade tiktoken
ARG TIKTOKEN_URL="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
RUN wget -O /opt/tiktoken_cache/$(echo -n $TIKTOKEN_URL | sha1sum | head -c 40) $TIKTOKEN_URL

ENTRYPOINT [ "python", "-m", "process" ]
