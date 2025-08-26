#!/usr/bin/env bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

# 스크립트가 위치한 디렉토리를 기준으로 경로를 설정
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKER_TAG="toothfairy3-multiinstance-algorithm-ensemble"

echo "Building Docker image: ${DOCKER_TAG}"
echo "Build context: ${SCRIPTPATH}"
echo "--------------------------------------------------"

# Docker Buildx를 사용하여 캐시 및 빌드 효율성 향상
# --progress=plain: 빌드 과정의 모든 로그를 상세히 보여줌
# --build-arg BUILDKIT_INLINE_CACHE=1: 빌드 캐시를 이미지에 포함시켜 다음 빌드 속도 향상
DOCKER_BUILDKIT=1 docker build "$SCRIPTPATH" \
    --platform=linux/amd64 \
    --progress=plain \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --tag "${DOCKER_TAG}"

echo "--------------------------------------------------"
echo "✅ Docker image built successfully: ${DOCKER_TAG}"