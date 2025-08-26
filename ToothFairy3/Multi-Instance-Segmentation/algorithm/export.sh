#!/usr/bin/env bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e
# 파이프라인(|)으로 연결된 명령어 중 하나라도 실패하면 전체를 실패로 간주
set -o pipefail

# 스크립트가 위치한 디렉토리를 기준으로 경로를 설정
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKER_TAG="toothfairy3-multiinstance-algorithm-ensemble"
OUTPUT_FILE="${SCRIPTPATH}/${DOCKER_TAG}.tar.gz"

echo "Exporting Docker image: ${DOCKER_TAG}"
echo "Saving to: ${OUTPUT_FILE}"
echo "--------------------------------------------------"
echo "This may take a few minutes..."

# docker save의 출력을 gzip으로 파이핑하여 압축 파일을 생성
docker save "${DOCKER_TAG}" | gzip -c > "${OUTPUT_FILE}"

echo "--------------------------------------------------"
echo "✅ Docker image exported successfully."

# ls -lh 명령어로 보기 쉬운 형태로 파일 크기를 함께 출력
ls -lh "${OUTPUT_FILE}"