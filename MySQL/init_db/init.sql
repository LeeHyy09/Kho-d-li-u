CREATE DATABASE IF NOT EXISTS weather_data;
USE weather_data;

CREATE TABLE hourly_weather_data (
    time Datetime PRIMARY KEY,                -- Thời gian theo chuẩn ISO8601
    temperature_2m FLOAT,                   -- Nhiệt độ ở độ cao 2m (°C)
    relative_humidity_2m FLOAT,             -- Độ ẩm tương đối ở độ cao 2m (%)
    dew_point_2m FLOAT,                     -- Điểm sương ở độ cao 2m (°C)
    apparent_temperature FLOAT,             -- Nhiệt độ cảm nhận (°C)
    rain FLOAT,                             -- Lượng mưa (mm)
    surface_pressure FLOAT,                 -- Áp suất mặt đất (hPa)
    cloud_cover FLOAT,                      -- Mức độ bao phủ mây tổng thể (%)
    wind_speed_10m FLOAT,                   -- Tốc độ gió ở độ cao 10m (km/h)
    wind_direction_10m FLOAT,               -- Hướng gió ở độ cao 10m (°)
    wind_gusts_10m FLOAT                    -- Gió giật ở độ cao 10m (km/h)
);
CREATE TABLE daily_weather_data (
    time Date PRIMARY KEY,                -- Thời gian theo chuẩn ISO8601
    temperature_2m FLOAT,                   -- Nhiệt độ ở độ cao 2m (°C)
    relative_humidity_2m FLOAT,             -- Độ ẩm tương đối ở độ cao 2m (%)
    dew_point_2m FLOAT,                     -- Điểm sương ở độ cao 2m (°C)
    apparent_temperature FLOAT,             -- Nhiệt độ cảm nhận (°C)
    rain FLOAT,                             -- Lượng mưa (mm)
    surface_pressure FLOAT,                 -- Áp suất mặt đất (hPa)
    cloud_cover FLOAT,                      -- Mức độ bao phủ mây tổng thể (%)
    wind_speed_10m FLOAT,                   -- Tốc độ gió ở độ cao 10m (km/h)
    wind_direction_10m FLOAT,               -- Hướng gió ở độ cao 10m (°)
    wind_gusts_10m FLOAT                    -- Gió giật ở độ cao 10m (km/h)
);
-- surface_pressure
-- wind_direction_10m
-- relative_humidity_2m

-- temperature_2m
CREATE TABLE weekly_weather_data (
    week Date PRIMARY KEY,                -- Thời gian theo chuẩn ISO8601
    temperature_2m FLOAT,                   -- Nhiệt độ ở độ cao 2m (°C)
    relative_humidity_2m FLOAT,             -- Độ ẩm tương đối ở độ cao 2m (%)
    dew_point_2m FLOAT,                     -- Điểm sương ở độ cao 2m (°C)
    apparent_temperature FLOAT,             -- Nhiệt độ cảm nhận (°C)
    rain FLOAT,                             -- Lượng mưa (mm)
    surface_pressure FLOAT,                 -- Áp suất mặt đất (hPa)
    cloud_cover FLOAT,                      -- Mức độ bao phủ mây tổng thể (%)
    wind_speed_10m FLOAT,                   -- Tốc độ gió ở độ cao 10m (km/h)
    wind_direction_10m FLOAT,               -- Hướng gió ở độ cao 10m (°)
    wind_gusts_10m FLOAT                    -- Gió giật ở độ cao 10m (km/h)
);
CREATE TABLE monthly_weather_data (
    month Date PRIMARY KEY,                -- Thời gian theo chuẩn ISO8601
    temperature_2m FLOAT,                   -- Nhiệt độ ở độ cao 2m (°C)
    relative_humidity_2m FLOAT,             -- Độ ẩm tương đối ở độ cao 2m (%)
    dew_point_2m FLOAT,                     -- Điểm sương ở độ cao 2m (°C)
    apparent_temperature FLOAT,             -- Nhiệt độ cảm nhận (°C)
    rain FLOAT,                             -- Lượng mưa (mm)
    surface_pressure FLOAT,                 -- Áp suất mặt đất (hPa)
    cloud_cover FLOAT,                      -- Mức độ bao phủ mây tổng thể (%)
    wind_speed_10m FLOAT,                   -- Tốc độ gió ở độ cao 10m (km/h)
    wind_direction_10m FLOAT,               -- Hướng gió ở độ cao 10m (°)
    wind_gusts_10m FLOAT                    -- Gió giật ở độ cao 10m (km/h)
);
CREATE TABLE weather_trend_residual (
    date DATETIME PRIMARY KEY,
    temperature FLOAT NULL,
    trend FLOAT NULL,
    residual FLOAT NULL
);
CREATE TABLE weather_seasonal (
    month date PRIMARY KEY,
    temperature FLOAT NULL, 
    seasonal FLOAT NULL
);
CREATE TABLE weather_season (
    date DATE PRIMARY KEY,
    temperature_2m FLOAT NOT NULL,
    kmean_label INT NOT NULL,
    half_year INT NOT NULL,
    adjusted_label INT NOT NULL
);
CREATE TABLE predicted_temperature (
    predicted_temp FLOAT
);

CREATE TABLE centroids (
    cluster_Centr INT,
    centroid FLOAT
);

CREATE TABLE probabilities (
    cluster INT,
    probability FLOAT
);

CREATE TABLE probability_summary (
    centroids FLOAT,
    cluster_Proba FLOAT,
    predicted_Temp FLOAT
);
CREATE TABLE prediction_temp(
    actual_temp FLOAT,
    predicted_temp FLOAT
);
