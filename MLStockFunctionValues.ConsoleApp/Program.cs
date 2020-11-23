using System;
using MLStockFunctionValuesML.Model;

namespace MLStockFunctionValues.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var predictClose = ModelBuilder.CreateModel("Close");
            var predictHigh = ModelBuilder.CreateModel("High");
            var predictLow = ModelBuilder.CreateModel("Low");
        }
    }
}
