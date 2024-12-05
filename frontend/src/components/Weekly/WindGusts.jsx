import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const WinGust = () => {
  const [timeData, setTimeData] = useState([]);
  const [winGustData, setWinGustData] = useState([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/weather/weekly');
        const times = response.data.map(item => new Date(item.week));
        const wind = response.data.map(item => item.wind_gusts_10m);
        setTimeData(times);
        setWinGustData(wind);
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

  if (timeData.length === 0 || winGustData.length === 0) {
    return <p>Đang tải dữ liệu...</p>;
  }

  return (
    <div>
      <h2>Wind Gusts 10m</h2>
      <span>Tốc độ gió giật mạnh nhất ở độ cao 10m trong một khoảng thời gian ngắn, đo bằng km/h.</span>
      <Plot
        data={[
          {
            x: timeData,
            y: winGustData,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: 'green', width: 2, shape: 'spline' },
            marker: { color: 'orange', size: 3, symbol: 'circle' },
          },
        ]}
        layout={{
          title: 'Wind Gusts vs Time (Weekly)',
          xaxis: {
            title: 'Time (Weekly)',
            type: 'date',
            tickformat: '%d/%m/%Y',
            tickvals: timeData.filter((_, index) => index % 3 === 0),
          },
          yaxis: {
            title: 'Wind Gusts (km/h)',
            rangemode: 'tozero',
          },
        }}
        style={{ width: '100%', height: '500px' }}
      />
    </div>
  );
};

export default WinGust;
