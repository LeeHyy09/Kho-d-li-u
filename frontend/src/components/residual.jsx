import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const Residual = () => {
  const [dateData, setDateData] = useState([]);
  const [residualData, setResidualData] = useState([]);

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/weather/pattern5')
      .then((response) => {
        const dates = response.data.map(item => new Date(item.date).toISOString());
        const residuals = response.data.map(item => item.residual);

        setDateData(dates);
        setResidualData(residuals);
      })
      .catch((error) => {
        console.error('Error fetching data:', error);
      });
  }, []);

  return (
    <div>

      <Plot
        data={[
          {
            x: dateData,
            y: residualData,
            type: 'scatter',
            mode: 'markers',
            marker: { color: 'red', size: 6, symbol: 'circle' },
          },
          {
            x: [dateData[0], dateData[dateData.length - 1]],
            y: [0, 0],
            type: 'scatter',
            mode: 'lines',
            line: { color: 'blue', width: 4, dash: 'dash' }, // Tăng độ dày (width: 4) và nét đứt (dash: 'dash')
          },
        ]}
        layout={{
          title: { text: 'Residual vs Time', font: { size: 24 } },
          xaxis: {
            title: { text: 'Date', font: { size: 18 } },
            type: 'date',
            tickformat: '%b %Y',
            tickangle: -45,
          },
          yaxis: {
            title: { text: 'Residual', font: { size: 18 } },
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

export default Residual;