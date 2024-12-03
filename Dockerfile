FROM jvizcar/braak-study:latest

# Install Python libraries for CZI file support.
USER root

RUN pip install imagecodecs[all]
RUN pip install czifile

USER myuser