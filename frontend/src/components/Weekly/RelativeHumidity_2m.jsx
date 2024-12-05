import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const HumidityDaily = () => {
  const [timeData, setTimeData] = useState([]);
  const [HumidityData, setHumidityData] = useState([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/weather/weekly');
        const times = response.data.map(item => new Date(item.week)); // Chuyển đổi ngày
        const Humidity = response.data.map(item => item.relative_humidity_2m);
        setTimeData(times);
        setHumidityData(Humidity);
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

  if (timeData.length === 0 || HumidityData.length === 0) {
    return <p>Đang tải dữ liệu...</p>;
  }

  return (
    <div>
      <h2>Humidity</h2>
      <span> Độ ẩm tương đối đo ở độ cao 2m, đơn vị là phần trăm (%)</span>
      <Plot
        data={[
          {
            x: timeData,
            y: HumidityData,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: 'red', width: 2, shape: 'spline' },
            marker: { color: 'blue', size: 3, symbol: 'circle' },
          },
        ]}
        layout={{
          title: { text: 'Humidity vs Time (Weekly)', font: { size: 24 } },
          xaxis: {
            title: { text: 'Time', font: { size: 18 } },
            type: 'date', // Trục thời gian
            tickformat: '%d/%m/%Y', // Định dạng Ngày/Tháng/Năm
            tickvals: timeData.filter((_, index) => index % 3 === 0), // Hiển thị mỗi 5 ngày
            tickangle: -45, // Góc hiển thị
          },
          yaxis: {
            title: { text: 'Humidity (%)', font: { size: 18 } },
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

export default HumidityDaily;
