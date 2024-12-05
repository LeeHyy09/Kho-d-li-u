import requests
from datetime import datetime, timedelta
import pandas as pd
import mysql.connector
import logging
from sqlalchemy import create_engine

# Cấu hình logging
logging.basicConfig(filename='database_operations.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Hàm lấy dữ liệu từ API
def get_data_from_api():
    try:
        # Lấy ngày kết thúc là 1 ngày trước ngày hiện tại
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        # URL API
        url = f'https://archive-api.open-meteo.com/v1/archive?latitude=16.0678&longitude=108.22088&start_date=2020-01-01&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,rain,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m'
        # Gửi request
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        data = response.json()  # Chuyển dữ liệu thành JSON
        # Kiểm tra trường 'hourly'
        if 'hourly' in data:
            return data['hourly']  # Trả về dữ liệu hourly
        else:
            print("Không tìm thấy trường 'hourly' trong dữ liệu JSON.")
            return None
    except Exception as e:
        print(f"Error during request: {e}")
        return None

# 2. Hàm làm sạch dữ liệu hàng giờ
def clean_hourly_data(hourly_data):
    try:
        # Chuyển dữ liệu thành DataFrame
        dataRaw = pd.DataFrame(hourly_data)
        # Xử lý cột 'time' nếu có
        if 'time' in dataRaw.columns:
            # Chuyển đổi cột 'time' sang định dạng datetime
            dataRaw['time'] = pd.to_datetime(dataRaw['time'], format="%Y-%m-%dT%H:%M")
        return dataRaw
    except Exception as e:
        print(f"An error occurred while cleaning data: {e}")
        return None

# 3. Hàm tổng hợp dữ liệu hàng ngày
def aggregate_daily_data(hourly_data):
    try:
        # Chuyển dữ liệu thành DataFrame
        df = pd.DataFrame(hourly_data)
        
        # Đảm bảo cột 'time' đã ở định dạng datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        # Tạo cột ngày từ 'time'
        df['date'] = df['time'].dt.date
        
        # Tính trung bình các cột theo ngày (trừ cột 'time' và 'date')
        daily_weather_cleaned_df = df.groupby('date').mean(numeric_only=True).reset_index()
        
        return daily_weather_cleaned_df
    except Exception as e:
        print(f"An error occurred while aggregating daily data: {e}")
        return None

# 4. Hàm tổng hợp dữ liệu theo tuần
def aggregate_weekly_data(hourly_data):
    try:
        # Chuyển dữ liệu thành DataFrame
        df = pd.DataFrame(hourly_data)
        
        # Đảm bảo cột 'time' đã ở định dạng datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        # Tạo cột tuần từ 'time' (với tuần bắt đầu từ ngày chủ nhật)
        df['week'] = df['time'].dt.to_period('W').dt.start_time
        
        # Tính trung bình các cột theo tuần (trừ cột 'time' và 'week')
        weekly_weather_cleaned_df = df.groupby('week').mean(numeric_only=True).reset_index()
        
        return weekly_weather_cleaned_df
    except Exception as e:
        print(f"An error occurred while aggregating weekly data: {e}")
        return None

# 5. Hàm tổng hợp dữ liệu theo tháng
def aggregate_monthly_data(hourly_data):
    try:
        # Chuyển dữ liệu thành DataFrame
        df = pd.DataFrame(hourly_data)
        
        # Đảm bảo cột 'time' đã ở định dạng datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        # Tạo cột tháng từ 'time'
        df['month'] = df['time'].dt.to_period('M').dt.start_time
        
        # Tính trung bình các cột theo tháng (trừ cột 'time' và 'month')
        monthly_weather_cleaned_df = df.groupby('month').mean(numeric_only=True).reset_index()
        
        return monthly_weather_cleaned_df
    except Exception as e:
        print(f"An error occurred while aggregating monthly data: {e}")
        return None

# 6. Hàm lưu dữ liệu vào MySQL
def save_to_mysql(df, table_name, db_url):
    try:
        # Tạo engine kết nối tới MySQL
        engine = create_engine(db_url)
        # Lưu DataFrame vào bảng
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        print(f"Data successfully saved to table '{table_name}' in MySQL.")
    except Exception as e:
        print(f"Error while saving data to MySQL: {e}")

# 7. Hàm lưu dữ liệu vào bảng hourly_weather_data
def save_hourly_data_to_mysql(df, db):
    cursor = db.cursor()
    # Câu lệnh INSERT cho bảng hourly_weather_data
    insert_query = """
    INSERT INTO hourly_weather_data (
        time, temperature_2m, relative_humidity_2m, dew_point_2m, apparent_temperature,
        rain, surface_pressure, cloud_cover, wind_speed_10m, wind_direction_10m, wind_gusts_10m
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        cursor.execute("START TRANSACTION;")  # Bắt đầu giao dịch
        for index, row in df.iterrows():
            data = (
                row['time'], row['temperature_2m'], row['relative_humidity_2m'],
                row['dew_point_2m'], row['apparent_temperature'],
                row['rain'], row['surface_pressure'], row['cloud_cover'],
                row['wind_speed_10m'], row['wind_direction_10m'], row['wind_gusts_10m']
            )
            try:
                cursor.execute(insert_query, data)
                logging.info(f"Inserted record: {data}")
            except mysql.connector.Error as err:
                logging.error(f"Error inserting record: {data}, Error: {err}")
                db.rollback()
                break
        db.commit()  # Xác nhận giao dịch
        logging.info(f"{cursor.rowcount} records inserted.")
    except mysql.connector.Error as err:
        logging.error(f"General error: {err}")
        db.rollback()
    finally:
        cursor.close()

# 8. Quá trình xử lý chính
if __name__ == "__main__":
    # Kết nối tới MySQL
    db = mysql.connector.connect(
        user='root',
        password='',  # Thay bằng mật khẩu nếu cần
        host='localhost',
        database='weather_data'
    )
    
    # Lấy dữ liệu từ API
    hourly_data = get_data_from_api()
    if hourly_data:
        # Làm sạch dữ liệu hàng giờ
        weather_cleaned_df = clean_hourly_data(hourly_data)
        if weather_cleaned_df is not None:
            # Lưu dữ liệu vào bảng hourly_weather_data
            save_hourly_data_to_mysql(weather_cleaned_df, db)
            
            # Tổng hợp dữ liệu hàng ngày
            daily_weather_cleaned_df = aggregate_daily_data(weather_cleaned_df)
            if daily_weather_cleaned_df is not None:
                # Lưu dữ liệu tổng hợp hàng ngày
                db_url = "mysql+pymysql://root:@localhost/weather_data"
                save_to_mysql(daily_weather_cleaned_df, "daily_weather_data", db_url)
            else:
                print("Không thể tổng hợp dữ liệu hàng ngày.")
            
            # Tổng hợp dữ liệu theo tuần
            weekly_weather_cleaned_df = aggregate_weekly_data(weather_cleaned_df)
            if weekly_weather_cleaned_df is not None:
                # Lưu dữ liệu tổng hợp theo tuần
                save_to_mysql(weekly_weather_cleaned_df, "weekly_weather_data", db_url)
            else:
                print("Không thể tổng hợp dữ liệu theo tuần.")
            
            # Tổng hợp dữ liệu theo tháng
            monthly_weather_cleaned_df = aggregate_monthly_data(weather_cleaned_df)
            if monthly_weather_cleaned_df is not None:
                # Lưu dữ liệu tổng hợp theo tháng
                save_to_mysql(monthly_weather_cleaned_df, "monthly_weather_data", db_url)
            else:
                print("Không thể tổng hợp dữ liệu theo tháng.")
        else:
            print("Dữ liệu hàng giờ không hợp lệ.")
    else:
        print("Không thể lấy dữ liệu từ API.")
    
    db.close()
