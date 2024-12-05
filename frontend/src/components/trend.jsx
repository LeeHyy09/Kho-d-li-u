import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const Trend = () => {
  const [dateData, setDateData] = useState([]);
  const [trendData, setTrendData] = useState([]);

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/weather/pattern2')
      .then((response) => {
        const dates = response.data.map(item => new Date(item.date).toISOString());
        const trendValues = response.data.map(item => item.trend);

        setDateData(dates);
        setTrendData(trendValues);
      })
      .catch((error) => {
        console.error('Error fetching data:', error);
      });
  }, []);

  return (
    <div style={{ color: '#333',
      minHeight: '100vh',
      padding: '20px', }}>

      <Plot
        data={[
          {
            x: dateData,
            y: trendData,
            type: 'scatter',
            mode: 'lines',
            line: { color: 'yellow', width: 2 }, // Đường xu hướng màu vàng
          },
        ]}
        layout={{
          title: { text: 'Trend vs Time', font: { size: 24 } },
          xaxis: {
            title: { text: 'Date', font: { size: 18 } },
            type: 'date',
            tickformat: '%b %Y',
            tickangle: -45,
          },
          yaxis: {
            title: { text: 'Trend', font: { size: 18 } },
            range: [24, 28], // Giới hạn trục Y trong khoảng 24-26
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

export default Trend;