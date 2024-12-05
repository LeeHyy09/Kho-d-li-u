# Sử dụng base image của Python
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /weather_pipeline

# Sao chép file requirements.txt vào container và cài đặt dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Sao chép các file cần thiết vào container
COPY . .

# Chạy cả hai tệp Python trong một lệnh
CMD ["bash", "-c", "python getData.py && python setData.py"]

# http://localhost:8001.
