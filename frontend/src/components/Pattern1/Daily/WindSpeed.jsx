import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const WindSpeed = () => {
  const [timeData, setTimeData] = useState([]);
  const [windSpeedData, setWindSpeedData] = useState([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/weather/daily');
        const times = response.data.map(item => new Date(item.date)); 
        const windSpeed = response.data.map(item => item.wind_speed_10m);
        setTimeData(times);
        setWindSpeedData(windSpeed);
      } catch (error) {
        console.error('Error fetching data:', error);
        setError(true);
      }
    };

    fetchData();
  }, []);

  if (error) {
    return <p>Không thể tải dữ liệu, vui lòng thử lại sau.</p>;
  }

  if (timeData.length === 0 || windSpeedData.length === 0) {
    return <p>Đang tải dữ liệu...</p>;
  }

  return (
    <div>
      <h2>Wind Speed 10m (m/s)</h2>
      <span>Tốc độ gió ở độ cao 100m, đơn vị là km/h.</span>
      <Plot
        data={[
          {
            x: timeData,
            y: windSpeedData,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: 'purple', width: 2, shape: 'spline' },
            marker: { color: 'yellow', size: 3, symbol: 'circle' },
          },
        ]}
        layout={{
          title: 'Wind Speed vs Time(Daily)',
          xaxis: {
            title: 'Time (Daily)',
            type: 'date',
            tickformat: '%d/%m/%Y',
          },
          yaxis: {
            title: 'Wind Speed (m/s)',
            rangemode: 'tozero',
          },
          margin: { t: 50, l: 50, r: 50, b: 100 },
        }}
        style={{ width: "100%", height: "500px" }}
      />
    </div>
  );
};

export default WindSpeed;
