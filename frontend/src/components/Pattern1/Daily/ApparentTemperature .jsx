import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const ApparentTemperature = () => {
  const [timeData, setTimeData] = useState([]);
  const [apparentTemperatureData, setApparentTemperatureData] = useState([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/weather/daily');
        const times = response.data.map(item => new Date(item.date)); 
        const apparentTemperature = response.data.map(item => item.apparent_temperature);
        console.log(times)
        setTimeData(times);
        setApparentTemperatureData(apparentTemperature);
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

  if (timeData.length === 0 || apparentTemperatureData.length === 0) {
    return <p>Đang tải dữ liệu...</p>;
  }

  return (
    <div>
      <h2>Apparent Temperature</h2>
      <span>Nhiệt độ cảm nhận được (apparent temperature) là một chỉ số đo cảm giác nhiệt độ mà cơ thể con người nhận thấy, tính toán dựa trên các yếu tố như nhiệt độ không khí và độ ẩm.</span>
      <Plot
        data={[
          {
            x: timeData,
            y: apparentTemperatureData,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: 'blue', width: 2, shape: 'spline' },
            marker: { color: 'red', size: 3, symbol: 'circle' },
          },
        ]}
        layout={{
          title: 'Apparent Temperature vs Time (Daily)',
          xaxis: {
            title: 'Time (Daily)',
            type: 'date',
            tickformat: '%d/%m/%Y',
            tickvals: timeData.filter((_, index) => index % 60 === 0),
          },
          yaxis: {
            title: 'Apparent Temperature (°C)',
            rangemode: 'tozero',
          },
        }}
        style={{ width: '100%', height: '500px' }}
      />
    </div>
  );
};

export default ApparentTemperature;
