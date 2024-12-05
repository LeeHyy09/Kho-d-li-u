import React, { useState, useEffect } from 'react';
import axios from 'axios';  // Import axios để sử dụng
import Plot from 'react-plotly.js';  // Import thư viện Plotly để vẽ biểu đồ

// import './CorrelationMatrix.css';  // Import CSS để trang trí giao diện

const CorrelationMatrix = () => {
  const [correlationMatrix, setCorrelationMatrix] = useState(null);

  useEffect(() => {
    // Lấy dữ liệu từ API để tính toán ma trận correlation
    axios.get('http://127.0.0.1:5000/api/weather/correlation')  // Giả sử API này trả về ma trận correlation
      .then((response) => {
        const corrMatrix = response.data.correlation; // Dữ liệu ma trận correlation từ API
        setCorrelationMatrix(corrMatrix);
      })
      .catch((error) => {
        console.error('Error fetching correlation matrix:', error);
      });
  }, []);

  // Nếu ma trận correlation chưa được tải, hiển thị thông báo loading
  if (!correlationMatrix) {
    return <div>Loading...</div>;
    
  }

  // Vẽ biểu đồ heatmap cho ma trận correlation
  return (
    <div style={{ color: '#333',
      minHeight: '100vh',
      padding: '20px', }}>
      <h3>Ma Trận Correlation Giữa Các Biến</h3>
      <Plot
        data={[
          {
            z: correlationMatrix,  // Dữ liệu ma trận correlation
            type: 'heatmap',
            colorscale: 'RdBu', // Màu sắc biểu đồ
            colorbar: {
              title: 'Correlation',
            },
          },
        ]}
        layout={{
          title: { text: 'Correlation Matrix of Weather Variables', font: { size: 24 } },
          xaxis: {
            title: { text: 'Variables', font: { size: 18 } },
          },
          yaxis: {
            title: { text: 'Variables', font: { size: 18 } },
          },
          margin: { t: 50, l: 50, r: 50, b: 100 },
          plot_bgcolor: '#f0f2f5',
          paper_bgcolor: '#ffffff',
        }}
        style={{ width: '100%', height: '500px' }}
      />
    </div>
  );
};

export default CorrelationMatrix;