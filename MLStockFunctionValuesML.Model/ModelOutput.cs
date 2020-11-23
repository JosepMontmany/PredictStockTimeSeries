using System;
using System.Collections.Generic;
using System.Text;

namespace MLStockFunctionValuesML.Model
{
    public class ModelOutput
    {
        public float[] ForecastedValue { get; set; }

        public float[] LowerBoundValue { get; set; }

        public float[] UpperBoundValue { get; set; }
    }
}
