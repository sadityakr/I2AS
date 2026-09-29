# The I2AS analysis image — where every recipe and analysis script runs.
#
# Built from a context that i2as.session.analysis_sandbox.stage_image_context()
# writes: this file plus ONLY i2as/analysis and the three core modules it may
# import (data_reader, events, exceptions). The Station, the Orchestrator, the
# session layer and the drivers are not in the image at all.
#
# Build it from the Settings dialog (Analysis → Build image), or:
#   python -m i2as.session.analysis_sandbox build-image
#
# A lab that needs more libraries (scikit-image, torch, ...) builds its own
# image FROM this one and names it in Settings → Analysis → Image:
#   FROM i2as-analysis:latest
#   RUN pip install --no-cache-dir scikit-image

FROM python:3.12-slim

# The worker's scientific stack. numpy and h5py read the run file; matplotlib
# draws figures (headless, MPLBACKEND=Agg is set at run time); scipy is what
# most fits reach for.
RUN pip install --no-cache-dir \
        "numpy>=1.24" \
        "h5py>=3.9" \
        "matplotlib>=3.7" \
        "scipy>=1.11"

COPY i2as /opt/i2as/i2as
ENV PYTHONPATH=/opt/i2as \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

# Never root inside the container. The sandbox passes --user uid:gid on
# POSIX hosts so files written into the output folder belong to the person.
RUN useradd --create-home --uid 1000 analysis
USER analysis

WORKDIR /work
CMD ["python", "-m", "i2as.analysis", "--help"]
