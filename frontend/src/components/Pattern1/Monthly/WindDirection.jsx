import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const WindDirection = () => {
  const [timeData, setTimeData] = useState([]);
  const [windDirectionData, setWindDirectionData] = useState([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/weather/monthly');
        const times = response.data.map(item => new Date(item.month)); 
        const windDirection = response.data.map(item => item.wind_direction_10m);
        setTimeData(times);
        setWindDirectionData(windDirection);
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

  if (timeData.length === 0 || windDirectionData.length === 0) {
    return <p>Đang tải dữ liệu...</p>;
  }

  return (
    <div>
      <h2>Wind Direction 10m (°)</h2>
      <span>Hướng gió ở độ cao 10m, đo theo độ (°) từ hướng Bắc (0°).</span>
      <Plot
        data={[
          {
            x: timeData,
            y: windDirectionData,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: 'blue', width: 2, shape: 'spline' },
            marker: { color: 'red', size: 5, symbol: 'circle' },
          },
        ]}
        layout={{
          title: 'Wind Direction vs Time (Monthly)',
          xaxis: {
            title: 'Time (Monthly)',
            type: 'datetime',
            tickvals: timeData.filter((_, index) => index % 3 === 0),
            tickformat: '%d/%m/%Y',
          },
          yaxis: {
            title: 'Wind Direction (°)',
            rangemode: 'tozero',
          },
          margin: { t: 50, l: 50, r: 50, b: 100 },
        }}
        style={{ width: "100%", height: "500px" }}
      />
    </div>
  );
};

export default WindDirection;
