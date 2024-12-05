import React, { useState } from 'react';
import { Nav, Tab, Container, Row, Col } from 'react-bootstrap';
import ChartComponentDaily from './Daily/ChartComponentDaily';
import ChartComponentMonthly from './Monthly/ChartComponentMonthly';
import ChartComponentWeekly from './Weekly/ChartComponentWeekly';

const LayoutPattern1 = () => {
  // State quản lý tab đang được chọn
  const [activeKey, setActiveKey] = useState('daily');

  return (
    <Container fluid style={{ backgroundColor: '#1e293b', color: '#ffffff', minHeight: '100vh', padding: '20px' }}>
      <Row>
        <Col>
          <h1 className="text-center mb-4">Thống kê thời tiết</h1>
          <Tab.Container activeKey={activeKey} onSelect={(key) => setActiveKey(key)}>
            {/* Điều hướng Tabs */}
            <Nav variant="pills" className="justify-content-center mb-4" style={{ gap: '10px' }}>
              <Nav.Item>
                <Nav.Link eventKey="daily" className="text-white" style={{ backgroundColor: '#374151', borderRadius: '5px' }}>
                  Theo Ngày
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="weekly" className="text-white" style={{ backgroundColor: '#374151', borderRadius: '5px' }}>
                  Theo Tuần
                </Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="monthly" className="text-white" style={{ backgroundColor: '#374151', borderRadius: '5px' }}>
                  Theo Tháng
                </Nav.Link>
              </Nav.Item>
            </Nav>

            {/* Nội dung từng tab */}
            <Tab.Content>
              <Tab.Pane eventKey="daily">
                <h3 className="text-center mb-3">Biểu đồ thống kê theo Ngày</h3>
                <ChartComponentDaily />
              </Tab.Pane>
              <Tab.Pane eventKey="weekly">
                <h3 className="text-center mb-3">Biểu đồ thống kê theo Tuần</h3>
                <ChartComponentWeekly />
              </Tab.Pane>
              <Tab.Pane eventKey="monthly">
                <h3 className="text-center mb-3">Biểu đồ thống kê theo Tháng</h3>
                <ChartComponentMonthly />
              </Tab.Pane>
            </Tab.Content>
          </Tab.Container>
        </Col>
      </Row>
    </Container>
  );
};

export default LayoutPattern1;
