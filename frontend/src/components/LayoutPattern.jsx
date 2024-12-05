import React from 'react';

import Residual from './residual';
import Seasonal from './Seasonal';
import Trend from './trend';
const LayoutPattern = () => {

  return (
      <div>
          <h1>Pattern 2 : Trend</h1>
          <Trend/>
          <h1>Pattern 3 : Seasional</h1>
          <Seasonal/>
          <h1>Pattern 4 : residual</h1>
          <Residual/>
          
      </div>
  );
};

export default LayoutPattern;