FROM python:3.12 AS build
WORKDIR /src
COPY . .
RUN pip wheel --no-cache-dir --no-deps -w /wheels .

FROM python:3.12-slim
LABEL authors="Matthias Blum (mblum@ebi.ac.uk)"
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
