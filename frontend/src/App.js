import React from 'react';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import { Box } from '@mui/material';
import ChartComponent from './ChartComponent'; // Đảm bảo rằng bạn đã import đúng
const weatherTheme = createTheme({
  palette: {
    mode: 'light', // Đổi từ 'dark' sang 'light' để hỗ trợ nền sáng
    background: {
      default: '#f5f5f5', // Màu nền sáng
      paper: '#ffffff', // Nền trắng cho các thành phần
    },
    primary: {
      main: '#1976d2', // Xanh bầu trời
    },
    secondary: {
      main: '#ffcc00', // Vàng ánh sáng
    },
    text: {
      primary: '#333333', // Văn bản chính màu tối
      secondary: '#666666', // Văn bản phụ màu xám
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
    h5: {
      fontWeight: 600,
      color: '#333333',
    },
    body1: {
      color: '#666666',
    },
  },
});
function App() {
  return (
    <ThemeProvider theme={weatherTheme}>
    <div style={{ width: '100%'}}>
      <CssBaseline />
      <Box
        sx={{
          maxWidth: '100%', // Rộng hơn mặc định
          // margin: '0 auto',   // Căn giữa
          // padding: '20px',    // Thêm khoảng cách
          backgroundColor: 'background.paper', // Màu nền sáng
          borderRadius: '8px', // Bo góc nhẹ
          boxShadow: '0 4px 8px rgba(0,0,0,0.1)', // Đổ bóng nhẹ
        }}
      >
        <ChartComponent />
      </Box>
    </div>
  </ThemeProvider>
  );
}

export default App;
