FROM python:3.8.6-slim-buster

# Install glpk (glpsol) and other shit:
RUN apt-get update && \
    apt-get install -y --no-install-recommends glpk-utils && \
    apt-get install -y libblas-dev  liblapack-dev libopenblas-dev && \
    apt-get install -y gfortran gcc g++ && \
    apt-get install -y --no-install-recommends libedit-dev build-essential

# Update paths (for glpk not sure if needed):
ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}

# Install python modules
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Get entrypoint (although it should be present when mounting)
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Setup sequence at run time
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "./my_script.py"]
