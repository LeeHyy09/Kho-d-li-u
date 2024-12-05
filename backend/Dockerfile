# Sử dụng base image của Python
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file requirements.txt vào container và cài đặt dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Sao chép các file cần thiết vào container
COPY . /app

# Chạy ứng dụng web
CMD ["python", "app.py"]

# http://localhost:8000
