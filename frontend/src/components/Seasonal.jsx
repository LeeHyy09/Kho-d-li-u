import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const Seasonal = () => {
  const [dateData, setDateData] = useState([]);
  const [seasonalData, setSeasonalData] = useState([]);

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/weather/pattern4')
      .then((response) => {
        const dates = response.data.map(item => new Date(item.month).toISOString()); // Convert to ISO format
        const seasonals = response.data.map(item => item.seasonal);

        setDateData(dates);
        setSeasonalData(seasonals);
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
            y: seasonalData,
            type: 'scatter',
            mode: 'lines',  // Only draw lines, no markers
            line: { color: 'darkblue', width: 4 },  // Set line color to dark blue and width to 4
            name: 'Seasonal Data',
          },
        ]}
        layout={{
          title: { text: 'Seasonal vs Time', font: { size: 24 } },
          xaxis: {
            title: { text: 'Date', font: { size: 18 } },
            type: 'date',
            tickformat: '%b %Y', // Format date as month-year
            tickangle: -45,
          },
          yaxis: {
            title: { text: 'Seasonal', font: { size: 18 } },
            rangemode: 'tozero',
            range: [-5, 5], // Set y-axis range based on seasonal data
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

export default Seasonal;