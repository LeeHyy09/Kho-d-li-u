import React from 'react';

import TemperatureTime from './Temperatureovertime';

import RainDaily from './RainDaily';
import HumidityDaily from './RelativeHumidity_2m';
import ApparentTemperature from './ApparentTemperature ';
import CloudCover from './CloudCover';
import DewPoint from './DewPoint';
import SurfacePressure from './SurfacePressure';
import WindDirection from './WindDirection';
import WindSpeed from './WindSpeed';
import WinGust from './WindGusts';
const ChartComponent = () => {

  return (
      <div>
          <TemperatureTime/>
          <RainDaily/>
          <HumidityDaily/>
          <ApparentTemperature/>
          <CloudCover/>
          <DewPoint/>
          <SurfacePressure/>
          <WindDirection/>
          <WindSpeed/>
          <WinGust/>
      </div>
  );
};

export default ChartComponent;
