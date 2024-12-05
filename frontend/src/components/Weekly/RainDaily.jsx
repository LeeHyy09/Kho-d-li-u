import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const RainDaily = () => {
  const [timeData, setTimeData] = useState([]); // Dữ liệu thời gian
  const [rainData, setRainData] = useState([]); // Dữ liệu lượng mưa
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/weather/weekly');
        const times = response.data.map(item => new Date(item.week)); // Chuyển đổi ngày
        const rains = response.data.map(item => item.rain); // Lấy dữ liệu lượng mưa
        setTimeData(times);
        setRainData(rains);
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

  if (timeData.length === 0 || rainData.length === 0) {
    return <p>Đang tải dữ liệu...</p>;
  }

  return (
    <div>
      <h2>Rain Daily</h2>
      <span>Lượng mưa (mm)</span>
      <Plot
        data={[
          {
            x: timeData,
            y: rainData,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: 'blue', width: 2, shape: 'spline' },
            marker: { color: 'red', size: 3, symbol: 'circle' },
          },
        ]}
        layout={{
          title: { text: 'Rain vs Time (Weekly)', font: { size: 24 } },
          xaxis: {
            title: { text: 'Time (Weekly)', font: { size: 18 } },
            type: 'date', // Trục thời gian
            tickformat: '%d/%m/%Y', // Định dạng Ngày/Tháng/Năm
            tickvals: timeData.filter((_, index) => index % 3 === 0), // Hiển thị mỗi 5 ngày một lần
            tickangle: -45, // Góc hiển thị các giá trị thời gian
          },
          yaxis: {
            title: { text: 'Rain (mm)', font: { size: 18 } },
            rangemode: 'tozero',
            // type: 'log',  // Sử dụng logarithmic scale cho trục y
            tickformat: '.4f',  // Hiển thị 4 chữ số thập phân
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

export default RainDaily;
