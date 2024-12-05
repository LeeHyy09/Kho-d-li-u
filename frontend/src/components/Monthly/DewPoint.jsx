import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const DewPoint = () => {
  const [timeData, setTimeData] = useState([]);
  const [dewPointData, setDewPointData] = useState([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/weather/monthly');
        const times = response.data.map(item => new Date(item.month)); 
        const dewPoint = response.data.map(item => item.dew_point_2m);
        setTimeData(times);
        setDewPointData(dewPoint);
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

  if (timeData.length === 0 || dewPointData.length === 0) {
    return <p>Đang tải dữ liệu...</p>;
  }

  return (
    <div>
      <h2>Dew Point Daily (°C)</h2>
      <span>Điểm sương là nhiệt độ mà không khí phải đạt được để hơi nước trong không khí ngưng tụ thành sương. Đo ở độ cao 2m và đơn vị là °C.</span>
      <Plot
        data={[
          {
            x: timeData,
            y: dewPointData,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: 'purple', width: 2, shape: 'spline' },
            marker: { color: 'green', size: 5, symbol: 'circle' },
          },
        ]}
        layout={{
          title: 'Dew Point vs Time (Monthly)',
          xaxis: {
            title: 'Time (Monthly)',
            type: 'datetime',
            tickvals: timeData.filter((_, index) => index % 3 === 0),
          },
          yaxis: {
            title: 'Dew Point (°C)',
            rangemode: 'tozero',
          },
          margin: { t: 50, l: 50, r: 50, b: 100 },
        }}
        style={{ width: "100%", height: "500px" }}
      />
    </div>
  );
};

export default DewPoint;
