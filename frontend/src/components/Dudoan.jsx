import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './Dudoan.css';

const DuDoan = () => {
  const [predictedTemperature, setPredictedTemperature] = useState(null);
  const [message, setMessage] = useState('');
  const [image, setImage] = useState('');
  const [dateData, setDateData] = useState([]);
  const [temperatureData, setTemperatureData] = useState([]);
  const [colorData, setColorData] = useState([]);
  const [yearlyData, setYearlyData] = useState({}); // Lưu dữ liệu nhóm theo năm

  useEffect(() => {
    // Lấy dữ liệu từ API pattern6 để vẽ seasonal chart
    axios.get('http://127.0.0.1:5000/api/weather/pattern6')
      .then((response) => {
        const dates = response.data.map(item => new Date(item.date).toISOString());
        const temperatures = response.data.map(item => item.temperature_2m);
        const colors = response.data.map(item => item.adjusted_label === 0 ? 'blue' : 'red');
        setDateData(dates);
        setTemperatureData(temperatures);
        setColorData(colors);
      })
      .catch((error) => {
        console.error('Error fetching data for chart:', error);
      });

    // Dự đoán nhiệt độ từ API pattern7
    axios.get('http://127.0.0.1:5000/api/weather/pattern7')
      .then((response) => {
        const temp = response.data[0].predicted_Temperature;
        setPredictedTemperature(temp);
        if (temp < 20) {
          setMessage('Trời lạnh, độ ẩm không khí thấp');
          setImage('/anh1.jpeg');
        } else if (temp >= 20 && temp <= 30) {
          setMessage('Thời tiết dễ chịu, không khí thoải mái');
          setImage('/anh2.jpeg');
        } else {
          setMessage('Trời nóng, độ ẩm cao');
          setImage('/anh3.jpeg');
        }
      })
      .catch((error) => {
        console.error('Error fetching predicted temperature:', error);
      });

    // Lấy dữ liệu Pie Chart theo năm
    axios.get('http://127.0.0.1:5000/api/weather/spiderChart')
      .then((response) => {
        const groupedData = response.data.reduce((acc, item) => {
          const year = new Date(item.date).getFullYear();
          const label = item.adjusted_label;
          if (!acc[year]) acc[year] = { 0: 0, 1: 0 };
          acc[year][label] += 1;
          return acc;
        }, {});
        setYearlyData(groupedData);
      })
      .catch((error) => {
        console.error('Error fetching Pie Chart data:', error);
      });
  }, []);

  const renderPieChart = (year, data) => {
    const labels = ['Mùa nắng', 'Mùa mưa'];
    const values = [data[0], data[1]];
  
    return (
      <div key={year} className="pie-chart">
        <Plot
          data={[
            {
              type: 'pie',
              labels: labels,
              values: values,
              marker: {
                colors: ['#1f77b4', '#ff7f0e'],
              },
              hole: 0.0, // Để không có lỗ ở giữa
            },
          ]}
          layout={{
            title: {
              text: `SpiderChart - ${year}`,
              x: 0.5, // Căn giữa tiêu đề
              xanchor: 'center', // Xác định tiêu đề sẽ căn giữa
            },
            showlegend: true,
          }}
          style={{ width: '150%', height: '600px' }} // Phóng to hình tròn lên
        />
      </div>
    );
  };
  
  return (
    <div style={{ color: '#333', minHeight: '100vh', padding: '20px' }}>
      {/* Biểu đồ Seasonal */}
      <div className="chart-container">
        <h3>Biểu đồ nhiệt độ qua các ngày</h3>
        <Plot
          data={[
            {
              x: dateData,
              y: temperatureData,
              type: 'scatter',
              mode: 'markers',
              marker: { color: colorData, size: 6 },
              name: 'Temperature Data',
            },
            {
              x: [dateData[dateData.length - 1], new Date().toISOString()],
              y: [temperatureData[temperatureData.length - 1], predictedTemperature],
              type: 'scatter',
              mode: 'lines+markers',
              line: { color: 'blue', width: 4, dash: 'dot' },
              name: 'Predicted Temperature',
            },
          ]}
          layout={{
            title: { text: 'Temperature Prediction for Tomorrow', font: { size: 24 } },
            xaxis: { title: { text: 'Date', font: { size: 18 } }, type: 'date', tickformat: '%b %Y', tickangle: -45 },
            yaxis: { title: { text: 'Temperature (°C)', font: { size: 18 } }, rangemode: 'tozero', range: [10, 45] },
            margin: { t: 50, l: 50, r: 50, b: 100 },
            plot_bgcolor: '#f0f2f5',
            paper_bgcolor: '#ffffff',
          }}
          style={{ width: '100%', height: '400px' }}
        />
      </div>

      {/* Dự đoán nhiệt độ */}
      <div className="prediction-box">
        <div className="temperature-info">
          <h2>Dự đoán nhiệt độ ngày mai</h2>
          {predictedTemperature && <h3>Nhiệt độ: {predictedTemperature.toFixed(2)}°C</h3>}
          <p>{message}</p>
        </div>
        <div className="image-container">
          <img src={image} alt="weather" className="weather-image" />
        </div>
      </div>

      {/* Biểu đồ Pie Chart theo năm */}
      <h3>Pattern 5: Biểu đổ SpiderChart qua các năm</h3>
      <div className="pie-charts-container">
        {Object.entries(yearlyData).map(([year, data]) => renderPieChart(year, data))}
      </div>
    </div>
  );
};

export default DuDoan;
