FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS source
ARG SQLGLOT_COMMIT=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
WORKDIR /opt/sqlglot
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://codeload.github.com/tobymao/sqlglot/tar.gz/${SQLGLOT_COMMIT}', '/tmp/sqlglot.tar.gz')" \
 && tar xzf /tmp/sqlglot.tar.gz --strip-components=1 \
 && rm /tmp/sqlglot.tar.gz
ENV PYTHONPATH=/opt/sqlglot SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1

FROM source AS profiler
RUN pip install --no-cache-dir pytest==9.2.0 pytest-timeout==2.4.0 duckdb==2.0.0 numpy==2.5.5 pandas==3.1.1 python-dateutil==4.0.0 pytz==2026.6 typing_extensions==4.16.0 psutil==7.2.2 setuptools==84.2.1 setuptools-scm==10.3.4 iniconfig==2.3.1 packaging==26.3 pluggy==1.7.0 Pygments==2.21.0 tzdata==2026.6 vcs-versioning==2.6.0
WORKDIR /app

FROM source AS native
COPY native-requirements.txt /tmp/
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/native-requirements.txt
WORKDIR /build
COPY ast_native.cpp ./
RUN mkdir /out && if [ "$(uname -m)" = x86_64 ]; then \
      python -m ziglang c++ -O3 -DNDEBUG -target x86_64-linux-gnu -mcpu=baseline -fPIC -shared -std=c++17 \
        $(python3-config --includes) ast_native.cpp -o "/out/_ast_native$(python3-config --extension-suffix)"; \
    fi

FROM profiler AS selector
WORKDIR /app
RUN python -m compileall -q -x '/site-packages/' /usr/local/lib/python3.13 /opt/sqlglot/sqlglot /opt/sqlglot/tests
COPY selector.py map_store.py patches.py schedule.py semantics.py coverage_map.py source_reader.py typing_keys.py default_profile.py declarations.py import_effects.py value_profile.py identity_profile.py heavy_profile.py replay.py tc_replay.py state_diff.py fixture_flow.py ./
COPY registry_context.py registry_readers.json slice_replay.py guard.py ./
COPY expression_additions.py expression_consumers.json ./
COPY fe_replay.py fe_fence.py build_fe_maps.py fe_baseline.json.gz fe_consumer.json.gz fe_lines.json.gz ./
COPY ds_verify.py ds_operations.py ds_origin.py ds_worker.py ds_sites.json.gz ./
COPY ds_rows.py ds_unread.py ds_imports.py ds_reads.json.gz fe_modules.json.gz ./
COPY fork_check.py import_state.py mn_reduced.py inert_files.py ./
COPY maps/ ./maps/
COPY --from=native /out/ ./
RUN python map_store.py \
 && PYTHONHASHSEED=0 python import_state.py --record /opt/sqlglot \
 && PYTHONHASHSEED=0 python mn_reduced.py --record /opt/sqlglot \
 && python fork_check.py --closure /app/fork_closure.json \
 && PYTHONHASHSEED=0 python ds_unread.py --record /app/ds_unread_baseline.json.gz \
 && python -m compileall -q -l /app
CMD ["python", "/app/selector.py"]
