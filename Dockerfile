FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

RUN apt-get update
# OpenCV를 위한 베이스(?) 라이브러리
RUN apt-get install -y libgl1 && apt-get install -y libglib2.0-0 

# 현재 경로에 있는 requirements.txt를 이미지 내 작업 디렉터리(workspace)에 복사
COPY requirements.txt /workspace/requirements.txt

# 작업 디렉터리 설정
WORKDIR /workspace

# 작업 디렉터리에서 requirements.txt 설치하고, 해당 파일 삭제 -> 볼륨 마운트할 때 똑같은 파일 있어서
RUN pip install -r requirements.txt && rm requirements.txt