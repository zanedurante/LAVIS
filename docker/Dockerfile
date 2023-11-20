# Build: docker build -t video-image-llm .
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

COPY environment.yaml .
RUN conda env update -n base --file environment.yaml
RUN pip install transformers==4.28