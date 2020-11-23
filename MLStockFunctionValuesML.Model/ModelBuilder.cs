using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using MLStockFunctionValuesML.Model;

namespace MLStockFunctionValuesML.Model
{
    public static class ModelBuilder
    {
        private static string TRAIN_DATA_FILEPATH = @"..\..\..\..\Data\TrainNDX_3.csv";
        public static string MODEL_FILE = Path.GetFullPath("MLModel.zip");

        private static MLContext mlContext = new MLContext(seed: 1);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="value">Close,High,Low,Open</param>
        public static ModelOutput CreateModel(string value)
        {
            // Load Data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: TRAIN_DATA_FILEPATH,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            var trainList = mlContext.Data.CreateEnumerable<ModelInput>(trainingDataView, true);
            int trainSize = trainList.Count();

            var testList = trainList.TakeLast(60);

            IDataView testDataView = mlContext.Data.LoadFromEnumerable(testList);

            var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
                                        outputColumnName: "ForecastedValue",
                                        inputColumnName: value,
                                        windowSize: 5,
                                        seriesLength: 60,
                                        trainSize: trainSize,
                                        horizon: 1,
                                        confidenceLevel: 0.95f,
                                        confidenceLowerBoundColumn: "LowerBoundValue",
                                        confidenceUpperBoundColumn: "UpperBoundValue");

            SsaForecastingTransformer forecaster = forecastingPipeline.Fit(trainingDataView);

            Evaluate(testDataView, forecaster, mlContext, value);

            var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);

            forecastEngine.CheckPoint(mlContext, MODEL_FILE);

            return Forecast(testList, 1, forecastEngine, mlContext, value);
        }

        static void Evaluate(IDataView testData, ITransformer model, MLContext mlContext, string value)
        {
            IDataView predictions = model.Transform(testData);

            IEnumerable<float> actual =
                        mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
                            .Select(observed =>
                            {
                                switch (value)
                                {
                                    case "Close":
                                        return observed.Close;
                                    case "High":
                                        return observed.High;
                                    case "Low":
                                        return observed.Low;
                                    default:
                                        return observed.Close;
                                }

                            });

            IEnumerable<float> forecast =
                        mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
                            .Select(prediction => prediction.ForecastedValue[0]);

            var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

            var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
            var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("---------------------");
            Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
            Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");


        }

        static ModelOutput Forecast(IEnumerable<ModelInput> testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext, string value)
        {
            ModelOutput forecast = forecaster.Predict();

            IEnumerable<string> forecastOutput =
                testData.TakeLast(horizon)
                    .Select((ModelInput values, int index) =>
                    {
                        string rentalDate = values.OpenTime.ToString();
                        float actualValue;
                        switch (value)
                        {
                            case "Close":
                                actualValue = values.Close;
                                break;
                            case "High":
                                actualValue = values.High;
                                break;
                            case "Low":
                                actualValue = values.Low;
                                break;
                            default:
                                actualValue = values.Close;
                                break;
                        }

                        float lowerEstimate = forecast.LowerBoundValue[index];
                        float estimate = forecast.ForecastedValue[index];
                        float upperEstimate = forecast.UpperBoundValue[index];
                        return $"Date: {rentalDate}\n" +
                        $"Actual {value}: {actualValue}\n" +
                        $"Forecast: {estimate}\n";
                     });

            Console.WriteLine("Stock Forecast");
            Console.WriteLine("---------------------");
            foreach (var prediction in forecastOutput)
            {
                Console.WriteLine(prediction);
            }

            return forecast;
        }

    }
}
