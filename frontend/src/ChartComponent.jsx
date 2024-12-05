import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
  Paper,
  Grid,
} from '@mui/material';
import { styled } from '@mui/system';
import LayoutPattern1 from './components/LayoutPattern1';
import Trend from './components/trend';
import Seasonal from './components/Seasonal';
import CorrelationMatrix from './components/correlation';
import DuDoan from './components/Dudoan';

// Custom styled components
const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  background: theme.palette.background.paper,
  boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.4)',
  borderRadius: '12px',
  margin: theme.spacing(2),
}));

const ChartComponent = () => {
  const [activePattern, setActivePattern] = useState('pat1');

  const handleMenuClick = (pattern) => {
    setActivePattern(pattern);
  };

  const renderActivePattern = () => {
    switch (activePattern) {
      case 'pat1':
        return (
          <StyledPaper>
            <Typography variant="h5">Pattern 1: Daylight Duration Over Time</Typography>
            <LayoutPattern1 />
          </StyledPaper>
        );
      case 'pat2':
        return (
          <StyledPaper>
            <Typography variant="h5">Pattern 2: Trend Analysis</Typography>
            <Trend />
          </StyledPaper>
        );
      case 'pat3':
        return (
          <StyledPaper>
            <Typography variant="h5">Pattern 3: Seasonal Trends</Typography>
            <Seasonal />
          </StyledPaper>
        );
      case 'pat4':
        return (
          <StyledPaper>
            <Typography variant="h5">Pattern 4: Correlation Matrix</Typography>
            <CorrelationMatrix />
          </StyledPaper>
        );
      case 'pat5':
        return (
          <StyledPaper>
            <Typography variant="h5">Pattern 7: Temperature Prediction</Typography>
            <DuDoan />
          </StyledPaper>
        );
      default:
        return null;
    }
  };

  return (
    <div style={{ width: '99%', margin: '0 auto' }}>
      <Box sx={{ minHeight: '100vh',
        background: 'linear-gradient(to right, #3ab5b0 0%, #3d99be 31%, #56317a 100%)',
         color: '#333',display: 'flex',
        flexDirection: 'column' }}>
        {/* AppBar Header */}
        <AppBar position="static" sx={{ backgroundColor: 'rgba(30, 41, 59, 0.95)' }}>
          <Toolbar>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              Weather Dashboard - Đà Nẵng
            </Typography>
            <Button color="inherit" onClick={() => handleMenuClick('pat1')}>
              Pattern 1
            </Button>
            <Button color="inherit" onClick={() => handleMenuClick('pat2')}>
              Pattern 2
            </Button>
            <Button color="inherit" onClick={() => handleMenuClick('pat3')}>
              Pattern 3
            </Button>
            <Button color="inherit" onClick={() => handleMenuClick('pat4')}>
              Pattern 4
            </Button>
            <Button color="inherit" onClick={() => handleMenuClick('pat5')}>
              Pattern 5 vs 7
            </Button>
          </Toolbar>
        </AppBar>


        <Box sx={{
      flexGrow: 1, // Đẩy footer xuống dưới
    }}>
        {/* Main Content */}
        <Container maxWidth={false} sx={{ width: '99%', margin: '20px auto', mt: 1, mb: 4, }}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              {renderActivePattern()}
            </Grid>
          </Grid>
        </Container>
        </Box>

        {/* Footer */}
        <Box
          component="footer"
          sx={{
            textAlign: 'center',
            background: 'rgba(30, 41, 59, 0.9)',
            color: '#c5c5c5',          
            textAlign: 'center',
            py: 3,
          }}
        >
          <Typography variant="body2">&copy; 2024 Weather Dashboard - Nhóm xinh gái</Typography>
        </Box>
      </Box>
    </div>
  );
};

export default ChartComponent;

