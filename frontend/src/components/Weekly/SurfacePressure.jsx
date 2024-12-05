import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const SurfacePressure = () => {
  const [timeData, setTimeData] = useState([]);
  const [surfacePressureData, setSurfacePressureData] = useState([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/weather/weekly');
        const times = response.data.map(item => new Date(item.week)); 
        const surfacePressure = response.data.map(item => item.surface_pressure);
        setTimeData(times);
        setSurfacePressureData(surfacePressure);
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

  if (timeData.length === 0 || surfacePressureData.length === 0) {
    return <p>Đang tải dữ liệu...</p>;
  }

  return (
    <div>
      <h2>Surface Pressure (hPa)</h2>
      <span>Áp suất khí quyển ở bề mặt, đơn vị là hPa (hectopascal). Đây là áp suất không khí tác động lên mặt đất.</span>
      <Plot
        data={[
          {
            x: timeData,
            y: surfacePressureData,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: 'green', width: 2, shape: 'spline' },
            marker: { color: 'orange', size: 3, symbol: 'circle' },
          },
        ]}
        layout={{
          title: 'Surface Pressure vs Time (Weekly)',
          xaxis: {
            title: 'Time (Weekly)',
            type: 'date',
            tickformat: '%d/%m/%Y',
            tickvals: timeData.filter((_, index) => index % 3 === 0),
          },
          yaxis: {
            title: 'Surface Pressure (hPa)',
            rangemode: 'tozero',
          },
          margin: { t: 50, l: 50, r: 50, b: 100 },
        }}
        style={{ width: "100%", height: "500px" }}
      />
    </div>
  );
};

export default SurfacePressure;
