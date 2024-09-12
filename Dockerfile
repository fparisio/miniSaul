FROM condaforge/miniforge3:latest

WORKDIR /app

COPY environment.yml .

RUN mamba env create -f environment.yml && \
    mamba clean -a -y && \
    mamba init bash

COPY . .

EXPOSE 8501

SHELL ["/bin/bash", "-c"]

CMD ["/bin/bash", "-c", "source activate minisaul_cpu && streamlit run miniSaul/app.py"]
