#!/usr/bin/env bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

# --- 변수 설정 ---
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKER_TAG="toothfairy3-multiinstance-algorithm-ensemble"

# 테스트에 사용할 데이터 경로
TEST_DATA_DIR="/home/psw/nnUNet/data/nnUNet_raw/ToothFairy_FnPnS_sample"
# ✅ [수정] 모델 경로는 프로젝트 내부의 nnunet_model 디렉토리로 지정합니다.
MODEL_DIR="${SCRIPTPATH}/nnunet_model"

# 입출력 폴더 경로
INPUT_DIR="${SCRIPTPATH}/test/input"
OUTPUT_DIR="${SCRIPTPATH}/test/output"
FINAL_RESULTS_DIR="${SCRIPTPATH}/test-results/algorithm-output"

# ⭐️ [추가] 테스트에 사용할 모델 폴드를 변수로 지정합니다. (예: 0과 1)
#    스크립트에 전달할 때 "0 1" 형태의 문자열이 됩니다.
FOLDS_TO_USE="0 1"

# --- 테스트 환경 준비 ---
echo "Preparing test environment..."
# 입출력 디렉터리 생성
mkdir -p "${INPUT_DIR}/images/cbct"
mkdir -p "${OUTPUT_DIR}/images/oral-pharyngeal-segmentation"
mkdir -p "${OUTPUT_DIR}/metadata"
mkdir -p "${FINAL_RESULTS_DIR}"

# 테스트 이미지 1개 복사
echo "Copying one test image for validation..."
first_file=$(ls "${TEST_DATA_DIR}/imagesTr"/*.nii.gz | head -1)

if [ -z "${first_file}" ]; then
    echo "Error: No test images found in ${TEST_DATA_DIR}/imagesTr"
    exit 1
fi

cp "${first_file}" "${INPUT_DIR}/images/cbct/001_0000.nii.gz"
echo "  -> $(basename "${first_file}")"

# --- Docker 실행 함수 ---
run_docker_container() {
    echo "Running Docker container..."
    docker run --rm \
        --name "toothfairy3-multiinstance-algorithm-ensemble" \
        --user "$(id -u):$(id -g)" \
        -e HOME=/home/user \
        -e MPLCONFIGDIR=/tmp/matplotlib \
        --memory=16g \
        --shm-size=8g \
        -v "${INPUT_DIR}":/input \
        -v "${OUTPUT_DIR}":/output \
        -v "${MODEL_DIR}":/opt/algorithm/nnunet_model \
        "$@" # 함수로 전달된 추가 인자(이미지 태그, GPU 옵션, 스크립트 인자 등)를 여기에 추가
}

# --- GPU 옵션 결정 및 컨테이너 실행 ---
# 사용 가능한 GPU 옵션을 결정
GPU_ARGS=""
if command -v nvidia-docker &> /dev/null; then
    echo "Using nvidia-docker for GPU support."
elif docker run --gpus all --rm hello-world &> /dev/null; then
    echo "Using --gpus all for GPU support."
    GPU_ARGS=""
else
    echo "GPU not available. Running on CPU."
fi

# ✅ [수정] 결정된 옵션으로 Docker 컨테이너 실행
#    Dockerfile의 ENTRYPOINT로 지정된 process_new.py에 인자를 전달합니다.
echo "Running container with folds: ${FOLDS_TO_USE}"
run_docker_container "${DOCKER_TAG}" ${GPU_ARGS} \
    --semseg_folds ${FOLDS_TO_USE}

# --- 결과 처리 ---
echo "Copying results to ${FINAL_RESULTS_DIR}"
# 출력 폴더의 모든 내용을 최종 결과 폴더로 복사
cp -r "${OUTPUT_DIR}/"* "${FINAL_RESULTS_DIR}/"

echo "--------------------------------------------------"
echo "✅ Algorithm test completed successfully!"
echo "Results are available in: ${FINAL_RESULTS_DIR}"