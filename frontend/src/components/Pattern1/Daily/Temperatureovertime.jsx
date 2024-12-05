import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const TemperatureTime = () => {
  const [timeData, setTimeData] = useState([]);
  const [temperatureData, setTemperatureData] = useState([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/weather/daily');
        const times = response.data.map(item => new Date(item.date)); // Chuyển đổi ngày
        const temperatures = response.data.map(item => item.temperature_2m);
        setTimeData(times);
        setTemperatureData(temperatures);
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

  if (timeData.length === 0 || temperatureData.length === 0) {
    return <p>Đang tải dữ liệu...</p>;
  }

  return (
    <div>
      <h2>Temperature Daily</h2>
      <span>Nhiệt độ, đơn vị là độ C (°C)</span>
      <Plot
        data={[
          {
            x: timeData,
            y: temperatureData,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: 'red', width: 2, shape: 'spline' },
            marker: { color: 'blue', size: 3, symbol: 'circle' },
          },
        ]}
        layout={{
          title: { text: 'Temperature vs Time (Daily)', font: { size: 24 } },
          xaxis: {
            title: { text: 'Time (Daily)', font: { size: 18 } },
            type: 'date', // Trục thời gian
            tickformat: '%d/%m/%Y', // Định dạng Ngày/Tháng/Năm
            tickvals: timeData.filter((_, index) => index % 60 === 0), // Hiển thị mỗi 60 ngày
            tickangle: -45, // Góc hiển thị
          },
          yaxis: {
            title: { text: 'Temperature (°C)', font: { size: 18 } },
            rangemode: 'tozero',
          },
          margin: { t: 50, l: 50, r: 50, b: 100 },
          plot_bgcolor: '#f0f2f5',
          paper_bgcolor: '#ffffff',
        }}
        style={{ width: "100%", height: "500px" }}
      />
    </div>
  );
};

export default TemperatureTime;
