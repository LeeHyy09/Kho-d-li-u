import mysql.connector
from mysql.connector import Error
import pandas as pd
from sqlalchemy import create_engine
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Kết nối MySQL
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='weather_data'
        )
        if connection.is_connected():
            print("Kết nối đến cơ sở dữ liệu MySQL thành công.")
            return connection
        else:
            print("Không thể kết nối đến cơ sở dữ liệu.")
            return None
    except Error as e:
        print(f"Error: {e}")
        return None

# Lấy dữ liệu từ MySQL
def get_data_from_mysql(query):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            result = cursor.fetchall()
            df = pd.DataFrame(result)
            cursor.close()
            connection.close()
            return df
        except Error as e:
            print(f"Error executing query: {e}")
            return None
    else:
        return None

# Hàm tạo kết nối SQLAlchemy
def create_db_engine(db_url):
    try:
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        print(f"Lỗi khi tạo kết nối cơ sở dữ liệu: {e}")
        return None

# Hàm thay thế NaN thành None (NULL trong MySQL)
def handle_nan_values(dataframe):
    return dataframe.where(pd.notna(dataframe), None)

# Hàm chèn dữ liệu vào MySQL
def insert_data_to_mysql(dataframe, table_name, db_url):
    engine = create_db_engine(db_url)
    if engine is None:
        return
    try:
        dataframe = handle_nan_values(dataframe)
        dataframe.reset_index(inplace=True)
        dataframe.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print(f"Dữ liệu đã được chèn thành công vào bảng '{table_name}' trong MySQL.")
    except Exception as e:
        print(f"Lỗi khi chèn dữ liệu vào MySQL: {e}")

# Phân rã dữ liệu nhiệt độ theo mô hình cộng (Pattern 2)
def decompose_weather_data(dataframe):
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe.set_index('date', inplace=True)
    dataframe = dataframe.asfreq('D')  # Đảm bảo tần suất hàng ngày
    decomposition = seasonal_decompose(dataframe['temperature_2m'], model='additive', period=365)
    dataframe['trend'] = decomposition.trend
    dataframe['residual'] = decomposition.resid
    dataframe = dataframe.dropna()
    return dataframe

def main_pattern_2():
    data_weather_daily = get_data_from_mysql("SELECT * FROM daily_weather_data")
    if data_weather_daily is not None:
        weather_pattern_2 = data_weather_daily[['date', 'temperature_2m']].copy()
        weather_pattern_2 = decompose_weather_data(weather_pattern_2)
        print(weather_pattern_2.head(5))

        db_url = "mysql+pymysql://root:@localhost/weather_data"
        insert_data_to_mysql(weather_pattern_2, 'weather_trend_residual', db_url)

# Phân rã dữ liệu mùa vụ (Pattern 3)
def decompose_weather_data_seasonal(dataframe):
    dataframe['month'] = pd.to_datetime(dataframe['month'])
    dataframe.set_index('month', inplace=True)
    decomposition = seasonal_decompose(dataframe['temperature_2m'], model='additive', period=12)
    dataframe['seasonal'] = decomposition.seasonal
    return dataframe

def main_pattern_3():
    data_weather_monthly = get_data_from_mysql("SELECT * FROM monthly_weather_data")
    if data_weather_monthly is not None:
        weather_pattern_2_seasonal = data_weather_monthly[['month', 'temperature_2m']].copy()
        weather_pattern_2_seasonal = decompose_weather_data_seasonal(weather_pattern_2_seasonal)
        print(weather_pattern_2_seasonal.head(5))

        db_url = "mysql+pymysql://root:@localhost/weather_data"
        insert_data_to_mysql(weather_pattern_2_seasonal, 'weather_seasonal', db_url)

# Phân tích chu kỳ theo mùa với KMeans (Pattern 4)
# Hàm phân tích chu kỳ theo mùa
def seasonal_cycle_analysis():
    data_weather_daily = get_data_from_mysql("SELECT * FROM daily_weather_data")
    # Bước 1: Chuẩn bị dữ liệu
    weather_pattern = data_weather_daily[['date', 'temperature_2m']]
    weather_pattern['date'] = pd.to_datetime(weather_pattern['date'])
    X_features = weather_pattern[['temperature_2m']].values

    # Bước 2: Áp dụng KMeans để phân cụm
    k = 2  # Số cụm
    kmeans = KMeans(n_clusters=k, random_state=42)
    weather_pattern['kmean_label'] = kmeans.fit_predict(X_features)

    # Bước 3: Lưu tọa độ centroids
    centroids = kmeans.cluster_centers_

    # Bước 4: Thêm cột nửa năm (half_year) và điều chỉnh nhãn cụm (adjusted_label)
    weather_pattern['half_year'] = weather_pattern['date'].dt.month.apply(lambda x: 1 if x < 9 else 2)
    weather_pattern['adjusted_label'] = weather_pattern.apply(
        lambda row: 0 if row['half_year'] == 1 and row['kmean_label'] in [0, 1]
        else (1 if row['half_year'] == 2 and row['kmean_label'] in [0, 1]
              else row['kmean_label']),
        axis=1
    )

    # Bước 5: Dự đoán nhiệt độ ngày mai
    def predict_next_day_temperature(weather_pattern):
        today_temp = weather_pattern.iloc[-1]['temperature_2m']
        today_cluster = weather_pattern.iloc[-1]['adjusted_label']
        cluster_data = weather_pattern[weather_pattern['adjusted_label'] == today_cluster]
        cluster_data['temp_change'] = cluster_data['temperature_2m'].diff()
        avg_temp_change = cluster_data['temp_change'].mean()
        return today_temp + avg_temp_change

    predicted_temp = predict_next_day_temperature(weather_pattern)

    # Bước 6: Tính xác suất thuộc các cụm
    def calculate_cluster_probabilities(predicted_temp, centroids):
        distances = np.linalg.norm(centroids - predicted_temp, axis=1)
        probabilities = 1 / distances
        probabilities /= probabilities.sum()
        return probabilities

    probabilities = calculate_cluster_probabilities(np.array([[predicted_temp]]), centroids)

    # Tạo các DataFrame cần thiết
    # 1. Predicted Temperature
    predicted_temp_df = pd.DataFrame({
        'predicted_Temperature': [predicted_temp]
    })

    # 2. Centroids
    centroids_df = pd.DataFrame({
        'cluster': range(1, len(centroids) + 1),
        'centroid': centroids.flatten()
    })

    # 3. Probabilities
    probabilities_df = pd.DataFrame({
        'cluster': range(1, len(probabilities) + 1),
        'probability': probabilities.flatten()
    })

    # 4. Probability Summary
    probability_summary_df = pd.DataFrame({
        'centroids': centroids.flatten(),
        'cluster Probabilities': probabilities.flatten(),
    })
    probability_summary_df['Predicted Temperature'] = predicted_temp

    # 5. Weather Season
    weather_season_df = weather_pattern[['date', 'temperature_2m', 'kmean_label', 'half_year', 'adjusted_label']]

    # Cấu hình URL kết nối tới MySQL
    db_url = "mysql+pymysql://root:@localhost/weather_data"  

    # Chèn dữ liệu vào MySQL
    insert_data_to_mysql(predicted_temp_df, 'predicted_temperature', db_url)
    insert_data_to_mysql(centroids_df, 'centroids', db_url)
    insert_data_to_mysql(probabilities_df, 'probabilities', db_url)
    insert_data_to_mysql(probability_summary_df, 'probability_summary', db_url)
    insert_data_to_mysql(weather_season_df, 'weather_season', db_url)

# Gọi hàm seasonal_cycle_analysis với dữ liệu weather
# Đảm bảo data_weather_daily đã được định nghĩa trước khi gọi hàm
# seasonal_cycle_analysis(data_weather_daily)



# Dự báo nhiệt độ hàng ngày (Pattern 5)


# Giả sử các hàm sau đã được định nghĩa:
# - get_data_from_mysql: lấy dữ liệu từ MySQL
# - forecast_daily_temperature: dự báo nhiệt độ theo ngày
# - insert_data_to_mysql: chèn dữ liệu vào MySQL
# - process_data: xử lý dữ liệu cho mô hình








if __name__ == "__main__":
    # Chạy các pattern
    print("Chạy pattern 2 (trend)...")
    main_pattern_2()

    print("Chạy pattern 3 (seasonal)...")
    main_pattern_3()

    print("Chạy pattern 4 (seasonal cycle)...")
    seasonal_cycle_analysis()

    # print("Chạy pattern 5 (forecasting)...")
    # main_pattern_5()
    def process_data(data, observed_size, overlap_size, predict_distance):
    
        """
        Xử lý dữ liệu đầu vào để tạo X và y cho mô hình.
    
    Args:
        data (DataFrame): Dữ liệu gốc.
        observed_size (int): Số ngày quan sát.
        overlap_size (int): Kích thước chồng lấn.
        predict_distance (int): Khoảng cách dự đoán.

    Returns:
        X (ndarray): Dữ liệu huấn luyện.
        y (ndarray): Nhãn mục tiêu.
    """
        s = data.values
        samples = int(len(s) / (observed_size - overlap_size))

        X = []
        y = []

        for i in range(samples):
            start_idx = i * (observed_size - overlap_size)
            end_idx = start_idx + observed_size
            target_idx = start_idx + observed_size + predict_distance

            # Kiểm tra giới hạn chỉ số
            if target_idx < len(s):
                segment = s[start_idx:end_idx]

            # Làm đầy nếu segment không đủ kích thước observed_size
                if len(segment) < observed_size:
                    mean_value = np.mean(segment, axis=0)
                    segment = np.vstack([segment, np.tile(mean_value, (observed_size - len(segment), 1))])

                X.append(segment)
                y.append(s[target_idx][-1])

    # Chuyển đổi sang numpy array
        X = np.stack(X)
        y = np.array(y)

        return X, y


    def train_and_predict(data, observed_size, overlap_size, predict_distance, test_size=0.3, random_state=42, idx=11):
        """
    Huấn luyện mô hình và thực hiện dự đoán.

    Args:
        data (DataFrame): Dữ liệu đầu vào.
        observed_size (int): Số ngày quan sát.
        overlap_size (int): Kích thước chồng lấn.
        predict_distance (int): Khoảng cách dự đoán.
        test_size (float): Tỷ lệ dữ liệu kiểm tra.
        random_state (int): Seed cho dữ liệu ngẫu nhiên.
        idx (int): Chỉ số mẫu cần kiểm tra.

    Returns:
        DataFrame: DataFrame chứa kết quả thực tế và dự đoán.
    """
    # Xử lý dữ liệu
        X, y = process_data(data, observed_size, overlap_size, predict_distance)

    # Reshape dữ liệu X
        X = X.reshape(X.shape[0], -1)

    # Chia dữ liệu train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Khởi tạo và huấn luyện mô hình
        regr = RandomForestRegressor(max_depth=2, random_state=random_state)
        regr.fit(X_train, y_train)

    # Đánh giá mô hình
        score = regr.score(X_test, y_test)
    # print(f'Model score: {score}')

    # Dự đoán trên dữ liệu kiểm tra
        single_data = X_test[idx]
        predicted_temp = regr.predict(single_data.reshape(1, -1))

    # Tạo DataFrame chứa kết quả thực tế và dự đoán
        results_df = pd.DataFrame({
            'actual_temp': [y_test[idx]],
            'predicted_temp': [predicted_temp[0]]
    })

        return results_df


    def insert_data_to_mysql(df, table_name, db_url):
        """
    Chèn dữ liệu vào MySQL.
    
    Args:
        df (DataFrame): Dữ liệu cần chèn.
        table_name (str): Tên bảng.
        db_url (str): URL kết nối MySQL.
    """
        engine = create_engine(db_url)
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)

    data_weather_daily = get_data_from_mysql("SELECT * FROM daily_weather_data")
# Gọi hàm với dữ liệu của bạn
# Giả sử data_weather_daily là một DataFrame đã được xác định từ trước
    prediction_df = data_weather_daily[['relative_humidity_2m', 'surface_pressure', 'temperature_2m']]
    observed_size = 7
    overlap_size = 1
    predict_distance = 1

    results_df = train_and_predict(
        prediction_df,
        observed_size,
        overlap_size,
        predict_distance,
        idx=11  # Chỉ lấy kết quả của mẫu với chỉ số idx=11
)

# Hiển thị DataFrame kết quả
# print(results_df)

# Cấu hình URL kết nối tới MySQL
    db_url = "mysql+pymysql://root:@localhost/weather_data"  
# Chèn dữ liệu vào MySQL
    insert_data_to_mysql(results_df, 'prediction_temp', db_url)


