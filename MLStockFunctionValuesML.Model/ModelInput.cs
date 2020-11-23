using Microsoft.ML.Data;
using System;

namespace MLStockFunctionValuesML.Model
{
    public class ModelInput
    {
        [ColumnName("OpenTime"), LoadColumn(0)]
        public DateTime OpenTime { get; set; }


        [ColumnName("Open"), LoadColumn(1)]
        public float Open { get; set; }


        [ColumnName("High"), LoadColumn(2)]
        public float High { get; set; }


        [ColumnName("Low"), LoadColumn(3)]
        public float Low { get; set; }


        [ColumnName("Close"), LoadColumn(4)]
        public float Close { get; set; }

        [ColumnName("TickVolume"), LoadColumn(5)]
        public float TickVolume { get; set; }

    }
}
