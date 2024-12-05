import React from 'react';

import TemperatureTime from './Temperatureovertime';

import RainDaily from './RainDaily';
import HumidityDaily from './RelativeHumidity_2m';
// import ApparentTemperatureMonthly from './ApparentTemperatureMonthly ';
import ApparentTemperatureMonthly from './ApparentTemperature';
import CloudCover from './CloudCover';
import DewPoint from './DewPoint';
import SurfacePressure from './SurfacePressure';
import WindDirection from './WindDirection';
import WindSpeed from './WindSpeed';
import WinGust from './WindGusts';
const ChartComponentMonthly = () => {

  return (
      <div>
          <TemperatureTime/>
          <RainDaily/>
          <HumidityDaily/>
          <ApparentTemperatureMonthly/>
          <CloudCover/>
          <DewPoint/>
          <SurfacePressure/>
          <WindDirection/>
          <WindSpeed/>
          <WinGust/>
      </div>
  );
};

export default ChartComponentMonthly;
