import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const CloudCover = () => {
  const [timeData, setTimeData] = useState([]);
  const [cloudCoverData, setCloudCoverData] = useState([]);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/weather/monthly');
        const times = response.data.map(item => new Date(item.month)); 
        const cloudCover = response.data.map(item => item.cloud_cover);
        setTimeData(times);
        setCloudCoverData(cloudCover);
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

  if (timeData.length === 0 || cloudCoverData.length === 0) {
    return <p>Đang tải dữ liệu...</p>;
  }

  return (
    <div>
      <h2>Cloud Cover</h2>
      <span>Mức độ che phủ mây trong bầu trời, đơn vị là phần trăm (%). Chỉ số này cho biết bao nhiêu phần trăm bầu trời bị mây che phủ.</span>
      <Plot
        data={[
          {
            x: timeData,
            y: cloudCoverData,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: 'green', width: 2, shape: 'spline' },
            marker: { color: 'red', size: 3, symbol: 'circle' },
          },
        ]}
        layout={{
          title: 'Cloud Cover vs Time (Monthly)',
          xaxis: {
            title: 'Time (Monthly)',
            type: 'datetime',
            tickvals: timeData.filter((_, index) => index % 3 === 0),
          },
          yaxis: {
            title: 'Cloud Cover (%)',
            rangemode: 'tozero',
          },
        }}
        style={{ width: '100%', height: '500px' }}
      />
    </div>
  );
};

export default CloudCover;
