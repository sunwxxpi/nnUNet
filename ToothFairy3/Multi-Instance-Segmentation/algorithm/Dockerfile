# 1. 베이스 이미지 설정
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 2. 환경 변수 및 빌드 인수 설정
ARG APP_USER="user"

# 3. 시스템 및 사용자 설정 (root 권한으로 실행)
RUN groupadd -r "${APP_USER}" && useradd -m --no-log-init -r -g "${APP_USER}" "${APP_USER}" && \
    mkdir -p \
    /opt/app \
    /input \
    /output/images/oral-pharyngeal-segmentation \
    /output/metadata \
    /tmp/nnUNet_raw \
    /tmp/nnUNet_preprocessed && \
    chmod -R 777 /output && \
    chown -R "${APP_USER}:${APP_USER}" /opt/app /input /output /tmp

# 4. 사용자 전환 및 환경 설정
# pip 설치 전에 non-root 사용자로 전환합니다.
USER "${APP_USER}"
WORKDIR /opt/app

# 'user' 컨텍스트에서 환경 변수를 설정합니다.
ENV PATH="/home/${APP_USER}/.local/bin:${PATH}" \
    nnUNet_raw="/tmp/nnUNet_raw" \
    nnUNet_preprocessed="/tmp/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm"

# 5. 파이썬 의존성 설치 ('user'로서 실행)
COPY --chown=user:user requirements.txt .
RUN python -m pip install --user -U pip pip-tools && \
    python -m pip install --user -r requirements.txt

# 6. 애플리케이션 및 모델 파일 복사
COPY --chown=user:user nnunet_model /opt/algorithm/nnunet_model
COPY --chown=user:user process_ensemble.py .

# 7. 컨테이너 실행 설정
ENTRYPOINT [ "python", "-m", "process_ensemble" ]